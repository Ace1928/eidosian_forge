import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
class TvpLoss(nn.Module):
    """
    This class computes the losses for `TvpForVideoGrounding`. The process happens in two steps: 1) we compute
    hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of matched
    ground-truth / prediction (supervise class and box).

    Args:
        losses (`List[str]`):
            List of all the losses to be applied.
    """

    def __init__(self, losses):
        super().__init__()
        self.loss_map = {'iou': self.loss_iou, 'distance': self.loss_distance, 'duration': self.loss_duration}
        for loss in losses:
            if loss not in self.loss_map:
                raise ValueError(f'Loss {loss} not supported')
        self.losses = losses

    def loss_iou(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the intersection over union.
        """
        inter = torch.min(candidates_end_time, end_time) - torch.max(candidates_start_time, start_time)
        union = torch.max(candidates_end_time, end_time) - torch.min(candidates_start_time, start_time)
        iou = 1 - inter.clamp(min=0) / union
        return iou

    def loss_distance(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the distance of mid points.
        """
        mid_candidates = torch.div(torch.add(candidates_start_time, candidates_end_time), 2.0)
        mid_groundtruth = torch.div(torch.add(start_time, end_time), 2.0)
        distance_diff = torch.div(torch.max(mid_candidates, mid_groundtruth) - torch.min(mid_candidates, mid_groundtruth), duration).clamp(min=0.2)
        return distance_diff

    def loss_duration(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the difference of duration.
        """
        duration_candidates = torch.sub(candidates_end_time, candidates_start_time)
        duration_groundtruth = torch.sub(end_time, start_time)
        duration_diff = torch.square(torch.div(torch.sub(duration_candidates, duration_groundtruth), duration))
        duration_diff = duration_diff.clamp(min=0.4)
        return duration_diff

    def forward(self, logits, labels):
        """
        This performs the loss computation.

        Args:
            logits (`torch.FloatTensor`):
                The output logits of head module.
            labels (`List[torch.FloatTensor]`):
                List of tensors ([start, end, duration]), which contains start time, end time of the video corresponding to the text, and also the duration.
        """
        duration, start_time, end_time = labels
        candidates = torch.mul(logits, duration)
        candidates_start_time, candidates_end_time = (candidates[:, 0].float(), candidates[:, 1].float())
        losses_dict = {}
        for loss in self.losses:
            losses_dict.update({loss: self.loss_map[loss](start_time, end_time, candidates_start_time, candidates_end_time, duration)})
        return losses_dict
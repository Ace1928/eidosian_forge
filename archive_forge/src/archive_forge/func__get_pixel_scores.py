from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
def _get_pixel_scores(self, encoded: torch.Tensor) -> torch.Tensor:
    """Based on the encoder output, compute the scores for each pixel of the image"""
    scores = self.relu(self.conv_score_a(encoded))
    scores = self.conv_score_b(scores)
    scores = nn.functional.softmax(scores, 1)[:, :-1]
    batch_size, _, height, width = scores.shape
    scores = scores.permute(0, 2, 3, 1).reshape(batch_size, height, width, 8, 8)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(batch_size, height * 8, width * 8)
    scores = simple_nms(scores, self.nms_radius)
    return scores
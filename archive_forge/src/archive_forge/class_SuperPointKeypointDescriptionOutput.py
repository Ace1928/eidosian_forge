from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
@dataclass
class SuperPointKeypointDescriptionOutput(ModelOutput):
    """
    Base class for outputs of image point description models. Due to the nature of keypoint detection, the number of
    keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of images,
    the maximum number of keypoints is set as the dimension of the keypoints, scores and descriptors tensors. The mask
    tensor is used to indicate which values in the keypoints, scores and descriptors tensors are keypoint information
    and which are padding.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed during training.
        keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
            Relative (x, y) coordinates of predicted keypoints in a given image.
        scores (`torch.FloatTensor` of shape `(batch_size, num_keypoints)`):
            Scores of predicted keypoints.
        descriptors (`torch.FloatTensor` of shape `(batch_size, num_keypoints, descriptor_size)`):
            Descriptors of predicted keypoints.
        mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
            Mask indicating which values in keypoints, scores and descriptors are keypoint information.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
    """
    loss: Optional[torch.FloatTensor] = None
    keypoints: Optional[torch.IntTensor] = None
    scores: Optional[torch.FloatTensor] = None
    descriptors: Optional[torch.FloatTensor] = None
    mask: Optional[torch.BoolTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
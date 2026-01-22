from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig
@dataclass
class XCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for video-text similarity.
        logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
            The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
            The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`XCLIPTextModel`].
        video_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The video embeddings obtained by applying the projection layer to the pooled output of
            [`XCLIPVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPVisionModel`].
        mit_output (`BaseModelOutputWithPooling`):
            The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_video: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    video_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    mit_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output', 'mit_output'] else getattr(self, k).to_tuple() for k in self.keys()))
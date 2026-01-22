import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
class SpeechT5EncoderWithSpeechPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Encoder that applies SpeechT5SpeechEncoderPrenet to convert the audio waveform data to
    hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.prenet = SpeechT5SpeechEncoderPrenet(config)
        self.wrapped_encoder = SpeechT5Encoder(config)
        self.post_init()

    def forward(self, input_values: torch.FloatTensor, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        hidden_states, attention_mask = self.prenet(input_values, attention_mask)
        outputs = self.wrapped_encoder(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return outputs
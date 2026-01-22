import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
def _hard_upsample(self, hidden_states, durations):
    """
        Repeats the time dimension of each sample in the batch based on the corresponding duration.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, *)`, *optional*):
                The sequence to repeat, where `*` is any number of sequence-specific dimensions including none.
            durations (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates how many times to repeat time segments.
        """
    if hidden_states.size(0) == 1:
        hidden_states = torch.repeat_interleave(hidden_states, durations.view(-1), dim=1)
    else:
        if hidden_states.shape[0] > 1 and self.training:
            logger.warning_once('`self.training=True` and you use batching. You lose parallelism during the hifigan\n                               forward pass because the samples are interleaved.')
        hidden_states = [torch.repeat_interleave(hidden_state, duration, dim=0) for hidden_state, duration in zip(hidden_states, durations)]
        hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)
    return hidden_states
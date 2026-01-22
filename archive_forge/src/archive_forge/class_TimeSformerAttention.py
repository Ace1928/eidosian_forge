import collections
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig
class TimeSformerAttention(nn.Module):

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.attention = TimesformerSelfAttention(config)
        self.output = TimesformerSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, output_attentions)
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
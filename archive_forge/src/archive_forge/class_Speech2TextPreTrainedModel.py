import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
class Speech2TextPreTrainedModel(PreTrainedModel):
    config_class = Speech2TextConfig
    base_model_prefix = 'model'
    main_input_name = 'input_features'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        for i in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]
        subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        bsz = attention_mask.size()[0]
        attention_mask = torch.zeros((bsz, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask[torch.arange(bsz, device=attention_mask.device), subsampled_lengths - 1] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).long()
        return attention_mask
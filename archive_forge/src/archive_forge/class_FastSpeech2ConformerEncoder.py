import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerEncoder(nn.Module):
    """
    FastSpeech2ConformerEncoder encoder module.

    Args:
        config (`FastSpeech2ConformerConfig`):
            FastSpeech2ConformerConfig instance.
        module_config (`dict`):
            Dictionary containing the encoder or decoder module configuration from the `FastSpeech2ConformerConfig`.
        use_encoder_input_layer (`bool`, *optional*, defaults to `False`):
            Input layer type.
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config, use_encoder_input_layer=False):
        super().__init__()
        self.embed = None
        if use_encoder_input_layer:
            self.embed = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=0)
        self.pos_enc = FastSpeech2ConformerRelPositionalEncoding(config, module_config)
        self.conformer_layers = nn.ModuleList([FastSpeech2ConformerEncoderLayer(config, module_config) for _ in range(module_config['layers'])])

    def forward(self, input_tensor: torch.LongTensor, attention_mask: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=False, return_dict: Optional[bool]=None):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
            `torch.Tensor`:
                Output tensor of shape `(batch, time, attention_dim)`.
        """
        feature_representation = input_tensor
        if self.embed is not None:
            feature_representation = self.embed(feature_representation)
        hidden_states, pos_emb = self.pos_enc(feature_representation)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for conformer_layer in self.conformer_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = conformer_layer(hidden_states, pos_emb, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)
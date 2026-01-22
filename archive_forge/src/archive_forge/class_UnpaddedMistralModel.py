from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.mistral.configuration_mistral import MistralConfig
class UnpaddedMistralModel(UnpaddedMistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedMistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = UnpaddedMistralRotaryEmbedding(config.hidden_size // config.num_attention_heads, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta)
        self.layers = nn.ModuleList([UnpaddedMistralDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = UnpaddedMistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, nz_input_ids: torch.Tensor, nz_position_ids: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        nz_hidden_states = self.embed_tokens(nz_input_ids)
        cos_sin = self.rotary_emb()
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                nz_hidden_states = self._gradient_checkpointing_func(decoder_layer.__call__, cos_sin, nz_hidden_states, nz_position_ids, cu_seqlens, max_seqlen)
            else:
                nz_hidden_states = decoder_layer(cos_sin, nz_hidden_states, nz_position_ids, cu_seqlens, max_seqlen)
        nz_hidden_states = self.norm(nz_hidden_states)
        return nz_hidden_states
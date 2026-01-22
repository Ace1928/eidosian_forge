import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.punica import add_lora, add_lora_slice, bgmv
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import split_tensor_along_last_dim
class SamplerWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: Sampler, hidden_size: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

    @property
    def logits_as_hidden_states(self):
        return self.base_layer.logits_as_hidden_states

    @property
    def vocab_size(self):
        return self.base_layer.vocab_size

    @property
    def org_vocab_size(self):
        return self.base_layer.org_vocab_size

    @property
    def include_gpu_probs_tensor(self):
        return self.base_layer.include_gpu_probs_tensor

    def create_lora_weights(self, max_loras: int, lora_config: LoRAConfig, model_config: Optional[PretrainedConfig]=None) -> None:
        if 32000 < self.base_layer.vocab_size > 33024:
            raise ValueError('When using LoRA, vocab size must be 32000 >= vocab_size <= 33024')
        self.lora_a_stacked = torch.zeros((max_loras, 1, lora_config.max_lora_rank, self.hidden_size), dtype=lora_config.lora_dtype, device=self.device)
        self.lora_b_stacked = torch.zeros((max_loras, 1, math.ceil(self.base_layer.vocab_size / lora_config.lora_vocab_padding_size) * lora_config.lora_vocab_padding_size, lora_config.max_lora_rank), dtype=lora_config.lora_dtype, device=self.device)
        self.embeddings_tensors = torch.full((max_loras, lora_config.lora_extra_vocab_size, self.hidden_size), fill_value=float('-inf'), dtype=self.dtype, device=self.device)
        self.indices = None
        self.indices_padded = None
        self.indices_len = None

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0
        self.embeddings_tensors[index] = float('-inf')

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor, embeddings_tensor: Optional[torch.Tensor]):
        self.reset_lora(index)
        self.lora_a_stacked[index, 0, :lora_a.shape[1], :lora_a.shape[0]].copy_(lora_a.T, non_blocking=True)
        self.lora_b_stacked[index, 0, :lora_b.shape[1], :lora_b.shape[0]].copy_(lora_b.T, non_blocking=True)
        if embeddings_tensor is not None:
            self.embeddings_tensors[index, :embeddings_tensor.shape[0], :embeddings_tensor.shape[1]] = embeddings_tensor

    def set_mapping(self, base_indices: torch.Tensor, sampler_indices: torch.Tensor, sampler_indices_padded: torch.Tensor, embeddings_indices: torch.Tensor, indices_len: List[int]):
        self.indices = sampler_indices
        self.indices_padded = sampler_indices_padded
        self.indices_len = indices_len

    def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor, embedding_bias: Optional[torch.Tensor]=None) -> torch.Tensor:
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_gather(logits)
        if logits is None:
            return None
        lora_logits = torch.empty(self.embeddings_tensors.shape[0] + 1, self.embeddings_tensors.shape[1], hidden_states.shape[0], dtype=self.embeddings_tensors.dtype, device=self.embeddings_tensors.device)
        torch.matmul(self.embeddings_tensors, hidden_states.T, out=lora_logits[:-1])
        lora_logits[-1] = float('-inf')
        lora_logits = lora_logits.mT
        lora_logits = lora_logits.reshape(lora_logits.shape[0] * lora_logits.shape[1], lora_logits.shape[2]).index_select(0, self.indices_padded[:self.indices_len[2]]).nan_to_num_(nan=float('-inf'), posinf=float('inf'), neginf=float('-inf'))
        logits[:, self.base_layer.org_vocab_size:self.base_layer.org_vocab_size + lora_logits.shape[1]] = lora_logits
        _apply_lora(hidden_states, self.lora_a_stacked, self.lora_b_stacked, self.indices[:self.indices_len[1]], logits)
        logits = logits[:, :self.base_layer.vocab_size]
        return logits

    def forward(self, *args, **kwargs):
        return type(self.base_layer).forward(self, *args, **kwargs)
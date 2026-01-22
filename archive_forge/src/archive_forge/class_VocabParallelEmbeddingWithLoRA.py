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
class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer

    def create_lora_weights(self, max_loras: int, lora_config: LoRAConfig, model_config: Optional[PretrainedConfig]=None) -> None:
        lora_vocab_start_idx = self.base_layer.org_vocab_size
        weights_idx = None
        if self.base_layer.vocab_end_index > lora_vocab_start_idx:
            weights_idx = max(lora_vocab_start_idx - self.base_layer.vocab_start_index, 0)
            self.embeddings_slice = (self.base_layer.vocab_start_index - self.base_layer.org_vocab_size + weights_idx, self.base_layer.vocab_end_index - self.base_layer.org_vocab_size)
            self.embeddings_weights = self.base_layer.weight.data[weights_idx:]
            self.embeddings_weights.fill_(0)
        else:
            self.embeddings_slice = None
            self.embeddings_weights = None
        self.embeddings_tensors = torch.zeros((max_loras, lora_config.lora_extra_vocab_size, self.base_layer.embedding_dim), dtype=self.base_layer.weight.dtype, device=self.base_layer.weight.device)
        self.lora_a_stacked = torch.zeros((max_loras, self.base_layer.org_vocab_size + lora_config.lora_extra_vocab_size, lora_config.max_lora_rank), dtype=lora_config.lora_dtype, device=self.base_layer.weight.device)
        self.lora_b_stacked = torch.zeros((max_loras, 1, self.base_layer.embedding_dim, lora_config.max_lora_rank), dtype=lora_config.lora_dtype, device=self.base_layer.weight.device)
        self.lora_a_stacked_2d = self.lora_a_stacked.view(self.lora_a_stacked.shape[0] * self.lora_a_stacked.shape[1], self.lora_a_stacked.shape[2])
        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None
        self.embeddings_indices = None

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0
        self.embeddings_tensors[index] = 0

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor, embeddings_tensor: Optional[torch.Tensor]):
        self.reset_lora(index)
        self.lora_a_stacked[index, :lora_a.shape[0], :lora_a.shape[1]].copy_(lora_a, non_blocking=True)
        self.lora_b_stacked[index, 0, :lora_b.shape[1], :lora_b.shape[0]].copy_(lora_b.T, non_blocking=True)
        if embeddings_tensor is not None:
            self.embeddings_tensors[index, :embeddings_tensor.shape[0], :embeddings_tensor.shape[1]].copy_(embeddings_tensor, non_blocking=True)
            if self.embeddings_slice is not None:
                embeddings = self.embeddings_tensors.view(self.embeddings_tensors.shape[0] * self.embeddings_tensors.shape[1], self.embeddings_tensors.shape[2])[self.embeddings_slice[0]:self.embeddings_slice[1]]
                self.embeddings_weights[:embeddings.shape[0]].copy_(embeddings)

    def set_mapping(self, base_indices: torch.Tensor, sampler_indices: torch.Tensor, sampler_indices_padded: torch.Tensor, embeddings_indices: torch.Tensor, indices_len: List[int]):
        self.indices = base_indices
        self.embeddings_indices = embeddings_indices
        self.indices_len = indices_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added_tokens_mask = x > self.base_layer.org_vocab_size - 1
        indices = self.embeddings_indices[1][:self.indices_len[3]].view_as(x)
        full_lora_a_embeddings = F.embedding(x + indices, self.lora_a_stacked_2d)
        indices = self.embeddings_indices[0][:self.indices_len[3]].view_as(x)
        full_output = self.base_layer.forward(x.add_(indices * added_tokens_mask))
        full_output_org = full_output
        if full_output.ndim == 3:
            full_output = full_output.view(full_output.shape[0] * full_output.shape[1], -1)
        if full_lora_a_embeddings.ndim == 3:
            full_lora_a_embeddings = full_lora_a_embeddings.view(full_lora_a_embeddings.shape[0] * full_lora_a_embeddings.shape[1], -1)
        bgmv(full_output, full_lora_a_embeddings, self.lora_b_stacked, self.indices[:self.indices_len[0]], 0, 1.0)
        return full_output.view_as(full_output_org)
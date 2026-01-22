import copy
import json
import logging
import math
import os
import re
from typing import (Any, Callable, Dict, Hashable, List, Optional, Tuple, Type)
import safetensors.torch
import torch
from torch import nn
from vllm.config import LoRAConfig
from vllm.utils import LRUCache, in_wsl
from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping, from_layer, from_layer_sampler
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.utils import parse_fine_tuned_lora_name, replace_submodule
class LoRAModelManager:
    """A manager that manages multiple LoRA-fine-tuned models."""

    def __init__(self, model: nn.Module, max_num_seqs: int, max_num_batched_tokens: int, vocab_size: int, lora_config: LoRAConfig):
        """Create a LoRAModelManager and adapter for a given model.

        Args:
            model: the model to be adapted.
            max_num_seqs: the maximum number of sequences model can run in a
                single batch.
            max_num_batched_tokens: the maximum number of tokens model can run
                in a single batch.
            vocab_size: the vocab size of the model.
            lora_config: the LoRA configuration.
        """
        self.lora_config = lora_config
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.lora_slots
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.lora_index_to_id: List[Optional[int]] = [None] * self.lora_slots
        self.vocab_size = vocab_size
        self.base_indices = torch.empty(self.max_num_batched_tokens, dtype=torch.long, device='cuda')
        self.sampler_indices = torch.empty(self.max_num_batched_tokens, dtype=torch.long, device='cuda')
        self.sampler_indices_padded = torch.empty(self.max_num_batched_tokens, dtype=torch.long, device='cuda')
        self.embeddings_indices = torch.empty(2, self.max_num_batched_tokens, dtype=torch.long, device='cuda')
        self.offsets = []
        self.indices_len = [None] * 4
        self.model: nn.Module = model
        if hasattr(self.model, 'supported_lora_modules'):
            self.supported_lora_modules = copy.deepcopy(self.model.supported_lora_modules)
            self.packed_modules_mapping = copy.deepcopy(self.model.packed_modules_mapping)
        self.packed_modules: Dict[str, List[str]] = {}
        self.modules: Dict[str, 'BaseLayerWithLoRA'] = {}
        self._registered_loras: Dict[int, LoRAModel] = {}
        self._active_loras: Dict[int, None] = {}
        self._last_mapping = None
        self._create_lora_modules()
        self.model.lora_manager = self

    @property
    def capacity(self) -> int:
        return self.lora_config.max_cpu_loras

    @property
    def lora_slots(self) -> int:
        return self.lora_config.max_loras

    def __len__(self) -> int:
        return len(self._registered_loras)

    def activate_lora(self, lora_id: int) -> bool:
        """Move LoRA into a GPU buffer to be used in the forward pass."""
        if lora_id in self._active_loras:
            return False
        first_free_slot = next(((i, lora_id) for i, lora_id in enumerate(self.lora_index_to_id) if lora_id is None), None)
        if first_free_slot is None:
            raise ValueError('No free lora slots')
        index, _ = first_free_slot
        self._active_loras[lora_id] = None
        lora_model = self._registered_loras[lora_id]
        logger.debug(f'Activating LoRA. int id: {lora_model.id}, slot index: {index}')
        self.lora_index_to_id[index] = lora_model.id
        for module_name, module in self.modules.items():
            module_lora = lora_model.get_lora(module_name)
            if module_lora:
                module_lora.optimize()
                module.set_lora(index, module_lora.lora_a, module_lora.lora_b, module_lora.embeddings_tensor)
            else:
                module.reset_lora(index)
        return True

    def _deactivate_lora(self, lora_id: int):
        try:
            index = self.lora_index_to_id.index(lora_id)
            self.lora_index_to_id[index] = None
        except ValueError:
            pass

    def deactivate_lora(self, lora_id: int) -> bool:
        """Remove a LoRA from a GPU buffer."""
        if lora_id in self._active_loras:
            self._deactivate_lora(lora_id)
            self._active_loras.pop(lora_id)
            return True
        return False

    def _add_lora(self, lora: LoRAModel) -> bool:
        self._create_merged_loras_inplace(lora)
        self._registered_loras[lora.id] = lora

    def add_lora(self, lora: LoRAModel) -> bool:
        """Add a LoRAModel to the manager CPU cache."""
        if lora.id not in self._registered_loras:
            if len(self._registered_loras) >= self.capacity:
                raise RuntimeError('No free LoRA slots.')
            self._add_lora(lora)
            return True
        return False

    def remove_lora(self, lora_id: int) -> bool:
        """Remove a LoRAModel from the manager CPU cache."""
        self.deactivate_lora(lora_id)
        return bool(self._registered_loras.pop(lora_id, None))

    def _set_lora_mapping(self, mapping: LoRAMapping) -> None:
        base_indices, sampler_indices, sampler_indices_padded, embeddings_indices, indices_len = convert_mapping(mapping, self.lora_index_to_id, self.lora_slots + 1, self.vocab_size, self.lora_config.lora_extra_vocab_size)
        self.base_indices[:base_indices.shape[0]].copy_(base_indices)
        self.sampler_indices[:sampler_indices.shape[0]].copy_(sampler_indices)
        self.sampler_indices_padded[:sampler_indices_padded.shape[0]].copy_(sampler_indices_padded)
        self.embeddings_indices[:embeddings_indices.shape[0], :embeddings_indices.shape[1]].copy_(embeddings_indices)
        self.indices_len[:] = indices_len

    def set_lora_mapping(self, lora_mapping: LoRAMapping) -> None:
        if self._last_mapping != lora_mapping:
            self._set_lora_mapping(lora_mapping)
        self._last_mapping = lora_mapping

    def list_loras(self) -> Dict[int, LoRAModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_loras)

    def get_lora(self, lora_id: int) -> Optional[LoRAModel]:
        return self._registered_loras.get(lora_id, None)

    def remove_all_loras(self) -> bool:
        """Remove all LoRAModels from the manager."""
        self._registered_loras.clear()
        self.lora_index_to_id = [None] * self.lora_slots
        self._active_loras.clear()

    def _create_lora_modules(self):
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name):
                continue
            new_module = replace_submodule(self.model, module_name, from_layer(module, self.lora_slots, self.lora_config, self.model.config))
            if 'lm_head' in module_name:
                sampler_module = self.model.get_submodule('sampler')
                new_module = replace_submodule(self.model, 'sampler', from_layer_sampler(sampler_module, module, self.lora_slots, self.lora_config, self.model.config))
            self.register_module(module_name, new_module)
            self._register_packed_modules(module_name)
            new_module.set_mapping(self.base_indices, self.sampler_indices, self.sampler_indices_padded, self.embeddings_indices, self.indices_len)

    def register_module(self, module_name: str, module: 'BaseLayerWithLoRA'):
        assert isinstance(module, BaseLayerWithLoRA)
        self.modules[module_name] = module

    def create_dummy_lora(self, lora_id: int, rank: int, embedding_modules: Optional[Dict[str, str]]=None) -> LoRAModel:
        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAModel(lora_id, rank, {})
        for module_name, module in self.model.named_modules():
            if not self._match_target_modules(module_name) or not isinstance(module, BaseLayerWithLoRA):
                continue
            parts = module_name.split('.')
            if module_name not in self.packed_modules:
                if parts[-1] in embedding_modules:
                    input_dim = module.base_layer.org_vocab_size + self.lora_config.lora_extra_vocab_size if hasattr(module.base_layer, 'org_vocab_size') else module.base_layer.weight.shape[1]
                    output_dim = module.base_layer.embedding_dim if hasattr(module.base_layer, 'embedding_dim') else module.base_layer.weight.shape[0]
                    embeddings_tensor_dim = module.base_layer.embedding_dim if hasattr(module.base_layer, 'embedding_dim') else module.base_layer.weight.shape[1]
                    lora = LoRALayerWeights.create_dummy_lora_weights(module_name, input_dim, output_dim, rank, module.lora_a_stacked.dtype, 'cpu', embeddings_tensor_dim=embeddings_tensor_dim)
                else:
                    lora = LoRALayerWeights.create_dummy_lora_weights(module_name, module.lora_a_stacked.shape[-1], module.lora_b_stacked.shape[-2], rank, module.lora_a_stacked.dtype, 'cpu')
                lora.optimize()
            else:
                parts = module_name.split('.')
                replacements = self.packed_modules_mapping[parts[-1]]
                subloras = []
                for i, r in enumerate(replacements):
                    lora = LoRALayerWeights.create_dummy_lora_weights(module_name + '.' + r, module.lora_a_stacked[i].shape[-1], module.lora_b_stacked[i].shape[-2], rank, module.lora_a_stacked[i].dtype, 'cpu')
                    lora.optimize()
                    subloras.append(lora)
                lora = PackedLoRALayerWeights.pack(subloras)
            model.loras[module_name] = lora
        return model

    def _match_target_modules(self, module_name: str):
        return any((re.match('.*\\.{target_module}$'.format(target_module=target_module), module_name) or target_module == module_name for target_module in self.supported_lora_modules))

    def _register_packed_modules(self, module_full_name: str) -> None:
        parts = module_full_name.split('.')
        module_name = parts[-1]
        replacements = self.packed_modules_mapping.get(module_name)
        if not replacements:
            return
        prefix = '.'.join(parts[:-1])
        self.packed_modules[module_full_name] = [prefix + '.' + r if prefix else r for r in replacements]

    def _create_merged_loras_inplace(self, lora_model: LoRAModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_loras = []
            has_replacement = False
            for r in new_module_names:
                lora = lora_model.get_lora(r)
                replacement_loras.append(lora)
                if lora:
                    has_replacement = True
            if not has_replacement:
                continue
            for i in range(len(replacement_loras)):
                if replacement_loras[i]:
                    continue
                replacement_loras[i] = None
            lora_model.loras[module_name] = PackedLoRALayerWeights.pack(replacement_loras)
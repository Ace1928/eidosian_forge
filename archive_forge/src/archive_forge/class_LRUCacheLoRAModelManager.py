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
class LRUCacheLoRAModelManager(LoRAModelManager):
    """A model manager that manages multiple LoRAs with LRU cache."""

    def __init__(self, model: nn.Module, max_num_seqs: int, max_num_batched_tokens: int, vocab_size: int, lora_config: LoRAConfig):
        super().__init__(model, max_num_seqs, max_num_batched_tokens, vocab_size, lora_config)
        self._registered_loras: LoRALRUCache = LoRALRUCache(self.capacity, self.deactivate_lora)
        self._active_loras: LoRALRUCache = LoRALRUCache(self.lora_slots, self._deactivate_lora)

    def list_loras(self) -> Dict[int, LoRAModel]:
        """List all registered LoRAModels."""
        return dict(self._registered_loras.cache)

    def add_lora(self, lora: LoRAModel) -> bool:
        """Add a LoRAModel to the manager."""
        if lora.id not in self._registered_loras:
            self._add_lora(lora)
            was_added = True
        else:
            self._registered_loras.touch(lora.id)
            was_added = False
        return was_added

    def activate_lora(self, lora_id: int) -> bool:
        if lora_id not in self._active_loras and len(self._active_loras) >= self.lora_slots:
            self._active_loras.remove_oldest()
        result = super().activate_lora(lora_id)
        self._active_loras.touch(lora_id)
        return result

    def remove_oldest_lora(self) -> bool:
        if len(self._registered_loras) > 0:
            self._registered_loras.remove_oldest()
            return True
        return False
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
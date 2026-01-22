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
def _register_packed_modules(self, module_full_name: str) -> None:
    parts = module_full_name.split('.')
    module_name = parts[-1]
    replacements = self.packed_modules_mapping.get(module_name)
    if not replacements:
        return
    prefix = '.'.join(parts[:-1])
    self.packed_modules[module_full_name] = [prefix + '.' + r if prefix else r for r in replacements]
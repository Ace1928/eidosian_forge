from __future__ import annotations
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Literal, Optional
import torch
from torch import nn
from tqdm import tqdm
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.utils import (
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
def _unload_and_optionally_merge(self, merge=True, progressbar: bool=False, safe_merge: bool=False, adapter_names: Optional[list[str]]=None):
    if merge:
        if getattr(self.model, 'quantization_method', None) == 'gptq':
            raise ValueError('Cannot merge LORA layers when the model is gptq quantized')
    key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
    desc = 'Unloading ' + ('and merging ' if merge else '') + 'model'
    for key in tqdm(key_list, disable=not progressbar, desc=desc):
        try:
            parent, target, target_name = _get_submodules(self.model, key)
        except AttributeError:
            continue
        with onload_layer(target):
            if hasattr(target, 'base_layer'):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                new_module = target.modules_to_save[target.active_adapter]
                if hasattr(new_module, 'base_layer'):
                    if merge:
                        new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    new_module = new_module.get_base_layer()
                setattr(parent, target_name, new_module)
    return self.model
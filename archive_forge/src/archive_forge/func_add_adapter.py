from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import remove_hook_from_submodules
from torch import nn
from transformers.utils import PushToHubMixin
from peft.tuners.mixed import COMPATIBLE_TUNER_TYPES
from .config import PeftConfig
from .peft_model import PeftModel
from .tuners import (
from .utils import PeftType, _set_adapter, _set_trainable
def add_adapter(self, adapter_name: str, peft_config: PeftConfig):
    _check_config_compatible(peft_config)
    try:
        self.peft_config[adapter_name] = peft_config
        self.base_model.inject_adapter(self, adapter_name)
    except Exception:
        if adapter_name in self.peft_config:
            del self.peft_config[adapter_name]
        raise
    self.set_modules_to_save(peft_config, adapter_name)
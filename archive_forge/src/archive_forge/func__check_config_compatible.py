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
def _check_config_compatible(peft_config: PeftConfig) -> None:
    if peft_config.peft_type not in COMPATIBLE_TUNER_TYPES:
        raise ValueError(f"The provided `peft_type` '{peft_config.peft_type.value}' is not compatible with the `PeftMixedModel`. Compatible types are: {COMPATIBLE_TUNER_TYPES}")
from __future__ import annotations
import re
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
from .layer import Conv2d, IA3Layer, Linear
@staticmethod
def _check_target_module_feedforward(ia3_config, key) -> bool:
    """
        A helper private method that checks if the target module `key` matches with a feedforward module specified in
        `ia3_config`
        """
    if isinstance(ia3_config.feedforward_modules, str):
        is_feedforward = bool(re.fullmatch(ia3_config.feedforward_modules, key))
    else:
        is_feedforward = any((key.endswith(target_key) for target_key in ia3_config.feedforward_modules))
    return is_feedforward
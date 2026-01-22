from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from typing import Any
import torch
from torch import nn
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
from .config import PolyConfig
from .layer import Linear, PolyLayer
@contextmanager
def _manage_pre_hooks(self, task_ids):
    """Context manager to handle the lifecycle of pre hooks."""
    handles = self._register_pre_hooks(task_ids)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
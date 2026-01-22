from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
def _check_initialized(self, mixed_precision=None, cpu=None):
    """Checks if a modification is trying to be made and the `AcceleratorState` has already been initialized"""
    if self.initialized:
        err = 'AcceleratorState has already been initialized and cannot be changed, restart your runtime completely and pass `{flag}` to `Accelerator()`.'
        if cpu and self.device.type != 'cpu':
            raise ValueError(err.format(flag='cpu=True'))
        if mixed_precision is not None and mixed_precision != self._mixed_precision and (self.distributed_type != DistributedType.DEEPSPEED):
            raise ValueError(err.format(flag=f"mixed_precision='{mixed_precision}'"))
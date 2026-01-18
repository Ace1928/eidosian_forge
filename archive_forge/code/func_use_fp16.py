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
@property
def use_fp16(self):
    warnings.warn("The `use_fp16` property is deprecated and will be removed in version 1.0 of Accelerate use `AcceleratorState.mixed_precision == 'fp16'` instead.", FutureWarning)
    return self._mixed_precision != 'no'
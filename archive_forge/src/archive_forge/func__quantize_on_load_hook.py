import functools
import logging
import math
import os
import warnings
from contextlib import ExitStack
from functools import partial
from types import ModuleType
from typing import Any, Callable, ContextManager, Literal, Optional, OrderedDict, Set, Tuple, Type, cast
import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import init
from torch.nn.modules.module import _IncompatibleKeys
from typing_extensions import Self, override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import (
from lightning_fabric.utilities.types import _DEVICE
def _quantize_on_load_hook(quantize_fn: Callable[[torch.Tensor], None], state_dict: OrderedDict, *_: Any) -> None:
    weight_key = next((name for name in state_dict if name.endswith('weight')), None)
    if weight_key is None:
        return
    weight = state_dict.pop(weight_key)
    quantize_fn(weight)
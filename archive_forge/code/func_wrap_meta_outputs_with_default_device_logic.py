import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
def wrap_meta_outputs_with_default_device_logic(self, r, func, flat_args, device):
    converter = self.fake_tensor_converter
    common_device = None
    has_scalar_only_inputs = False

    def wrap(e):
        nonlocal common_device
        nonlocal has_scalar_only_inputs
        if isinstance(e, torch.Tensor) and common_device is None:
            common_device, has_scalar_only_inputs = FakeTensor._find_common_device(func, flat_args)
        if self.is_our_fake(e):
            torch._check(e.device == common_device, lambda: f'FakeTensor is wrapped to wrong device, found {e.device}, expected {common_device}')
        if isinstance(e, torch.Tensor) and (not self.is_our_fake(e)) and (converter is not None):
            if has_scalar_only_inputs:
                return converter(self, e)
            else:
                return converter.from_meta_and_device(self, e, device or common_device)
        else:
            return e
    return tree_map(wrap, r)
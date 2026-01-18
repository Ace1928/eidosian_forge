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
def validate_and_convert_non_fake_tensors(self, func, converter, flat_args, args_spec):
    """
        Checks if the list of tensors are fake tensors.
        If not, try to convert them to fake tensors.
        Returns the original args, kwargs, and a flattened list of (args, kwargs) that are fake tensors.
        """
    flat_arg_fake_tensors = []

    def validate(x):
        if not isinstance(x, torch.Tensor):
            return x
        nonlocal flat_arg_fake_tensors
        if not self.is_our_fake(x):
            if torch.Tag.inplace_view in func.tags:
                args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                raise Exception(f"Can't call metadata mutating ops on non-Fake Tensor inputs. Found in {render_call(func, args, kwargs)}")
            if not self.allow_non_fake_inputs:
                if isinstance(x, FakeTensor) and x.fake_mode is not self:
                    raise AssertionError('Mixing fake modes NYI')
                args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                raise Exception(f"Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'. Found in {render_call(func, args, kwargs)}")
            x = converter(self, x)
        flat_arg_fake_tensors.append(x)
        return x
    validated_args = [validate(a) for a in flat_args]
    return (validated_args, flat_arg_fake_tensors)
import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import operator
import re
import sys
import types
from typing import List, NamedTuple, Optional, Union
import torch
from torch import SymInt
from torch._guards import GuardSource, TracingContext
from torch._ops import HigherOrderOperator
from torch._streambase import _EventBase, _StreamBase
from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.immutable_collections import immutable_list
from torch.nested._internal.nested_tensor import NestedTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.weak import TensorWeakRef
from .. import config, mutation_guard, replay_record, skipfiles, trace_rules
from ..allowed_functions import (
from ..device_interface import get_registered_device_interfaces
from ..exc import InternalTorchDynamoError, unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..side_effects import SideEffects
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
from .dicts import (
from .distributed import (
from .functions import (
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lazy import LazyVariableTracker
from .lists import (
from .misc import (
from .nn_module import FSDPManagedNNModuleVariable, UnspecializedNNModuleVariable
from .optimizer import OptimizerVariable
from .tensor import (
from .torch import torch_special_class_types, TorchVariable
from .torch_function import build_torch_function_fn, TensorWithTFOverrideVariable
from .user_defined import (
def update_dim2constraint(dim, constraint_range, debug_name):
    if dim in dim2constraint:
        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        old_constraint_range, old_debug_name = dim2constraint[dim]
        new_constraint_range = StrictMinMaxConstraint(vr=constraint_range.vr & old_constraint_range.vr, warn_only=False)
        if old_debug_name is not None:
            assert debug_name is None or debug_name == old_debug_name
            new_debug_name = old_debug_name
        else:
            new_debug_name = debug_name
        dim2constraint[dim] = (new_constraint_range, new_debug_name)
    else:
        dim2constraint[dim] = (constraint_range, debug_name)
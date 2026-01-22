import inspect
import logging
import math
import re
import types
from typing import Dict, List
from torch._streambase import _StreamBase
from ..guards import install_guard
import torch._C
import torch._refs
import torch.fx
import torch.nn
import torch.onnx.operators
from .. import config, polyfill, variables
from ..allowed_functions import torch_get_name
from ..device_interface import get_registered_device_interfaces
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..utils import (
from .base import VariableTracker
from .ctx_manager import (
from .distributed import is_constant_pg_functions, is_from_local, ProcessGroupVariable
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lists import ListVariable, TupleVariable
from .torch_function import can_dispatch_torch_function, dispatch_torch_function
class BaseTorchVariable(VariableTracker):
    """common base for all torch.* functions, classes, modules and other things"""

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(value, source=source)

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def reconstruct(self, codegen):
        name = torch_get_name(value, f'allowed_fn_{id(value)}')
        unique_var_name = '__' + re.sub('[^a-zA-Z0-9_]+', '_', name)
        return codegen.setup_globally_cached(unique_var_name, value, False)

    def as_proxy(self):
        return self.value

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def call_hasattr(self, tx, name):
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def can_constant_fold_through(self):
        if self.value in constant_fold_functions:
            return True
        return getattr(self.value, '__module__', None) == 'math'
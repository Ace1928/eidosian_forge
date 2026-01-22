import collections
import functools
import inspect
import sys
import textwrap
import types
import warnings
from typing import Dict, List, Set, Type
import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import (
from torch.nn import Module
class ConcreteTypeStore:
    type_store: Dict[Type[Module], List[torch._C.ConcreteModuleType]]
    methods_compiled: Set[torch._C.ConcreteModuleType]

    def __init__(self):
        self.type_store = {}
        self.methods_compiled = set()

    def get_or_create_concrete_type(self, nn_module):
        """Infer a ConcreteType from this `nn.Module` instance. Underlying JIT types are re-used if possible."""
        concrete_type_builder = infer_concrete_type_builder(nn_module)
        nn_module_type = type(nn_module)
        if nn_module_type not in self.type_store:
            self.type_store[nn_module_type] = []
        known_types = self.type_store[nn_module_type]
        for known_type in known_types:
            if known_type.equals(concrete_type_builder):
                return known_type
        concrete_type = concrete_type_builder.build()
        self.type_store[nn_module_type].append(concrete_type)
        return concrete_type
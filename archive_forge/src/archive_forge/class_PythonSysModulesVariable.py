import collections
import dataclasses
import functools
import inspect
import sys
from typing import Any, Dict, List, Optional
import torch
import torch.fx
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name, istensor, istype, iter_contains
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .tensor import TensorVariable
class PythonSysModulesVariable(VariableTracker):
    """Special case for sys.modules.

    Without this we will guard on the exact set of modules imported in the
    lifetime of the python program.
    """

    def python_type(self):
        return dict

    @staticmethod
    def reconstruct(self, codegen):
        codegen.extend_output([codegen.create_load_python_module(sys, True), codegen.create_load_attr('modules')])

    def call_method(self, tx, name, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]):
        from .builder import VariableBuilder
        if name == '__getitem__':
            return self.call_getitem(tx, *args, **kwargs)
        elif name == 'get':
            return self.call_get(tx, *args, **kwargs)
        elif name == '__contains__':
            return self.call_contains(tx, *args, **kwargs)
        real_dict = VariableBuilder(tx, self.source)(sys.modules)
        return real_dict.call_method(tx, name, args, kwargs)

    def _contains_helper(self, tx, key: VariableTracker):
        k = ConstDictVariable.get_key(key)
        has_key = k in sys.modules
        install_guard(self.make_guard(functools.partial(GuardBuilder.DICT_CONTAINS, key=k, invert=not has_key)))
        return (k, has_key)

    def call_contains(self, tx, key: VariableTracker):
        k, has_key = self._contains_helper(tx, key)
        return ConstantVariable.create(value=has_key)

    def call_get(self, tx, key: VariableTracker, default: Optional[VariableTracker]=None):
        from .builder import VariableBuilder
        k, has_key = self._contains_helper(tx, key)
        if has_key:
            return VariableBuilder(tx, GetItemSource(self.source, k))(sys.modules[k])
        if default is not None:
            return default
        return ConstantVariable.create(value=None)

    def call_getitem(self, tx, key: VariableTracker):
        from .builder import VariableBuilder
        k, has_key = self._contains_helper(tx, key)
        return VariableBuilder(tx, GetItemSource(self.source, k))(sys.modules[k])
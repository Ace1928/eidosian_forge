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
def _as_set_element(self, vt):
    from .base import VariableTracker
    from .misc import MethodWrapperVariable
    from .tensor import TensorVariable
    assert isinstance(vt, VariableTracker)
    if isinstance(vt, TensorVariable):
        fake_tensor = vt.as_proxy().node.meta.get('example_value')
        if fake_tensor is None:
            unimplemented('Cannot check Tensor object identity without its fake value')
        return SetVariable.SetElement(vt, fake_tensor)
    if isinstance(vt, ConstantVariable):
        return SetVariable.SetElement(vt, vt.value)
    if isinstance(vt, MethodWrapperVariable):
        return SetVariable.SetElement(vt, vt.as_python_constant())
    unimplemented(f'Sets with {type(vt)} NYI')
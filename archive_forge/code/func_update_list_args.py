import weakref
from typing import Dict, List
import torch
from ..decorators import mark_static_address
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import ListVariable
from .misc import GetAttrVariable
from .user_defined import UserDefinedObjectVariable
def update_list_args(self, tx, args, kwargs, py_args, py_kwargs):
    """Update the args and kwargs to the traced optimizer call"""
    for arg, py_arg in zip(args, py_args):
        if isinstance(arg, ListVariable) and all((isinstance(t, torch.Tensor) for t in py_arg)):
            tensor_vars = ListVariable([self.wrap_tensor(tx, t) for t in py_arg], mutable_local=MutableLocal())
            tx.replace_all(arg, tensor_vars)
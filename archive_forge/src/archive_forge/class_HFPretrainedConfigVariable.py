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
class HFPretrainedConfigVariable(VariableTracker):
    """
    Hack for HuggingFace PretrainedConfig
    """

    @staticmethod
    def is_matching_cls(cls):
        mod = sys.modules.get('transformers.configuration_utils')
        is_match = mod is not None and issubclass(cls, mod.PretrainedConfig)
        if is_match:
            _install_PretrainedConfig_patch()
        return is_match

    @classmethod
    def is_matching_object(cls, obj):
        return cls.is_matching_cls(type(obj))

    def __init__(self, obj, **kwargs):
        super().__init__(**kwargs)
        self.obj = obj
        assert self.is_matching_cls(type(obj))

    def var_getattr(self, tx, name: str) -> 'VariableTracker':
        from . import ConstantVariable
        return ConstantVariable.create(getattr(self.obj, name))

    def call_hasattr(self, tx, name: str) -> 'VariableTracker':
        return variables.ConstantVariable.create(hasattr(self.obj, name))
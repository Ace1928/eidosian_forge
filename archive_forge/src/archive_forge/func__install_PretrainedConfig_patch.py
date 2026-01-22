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
@functools.lru_cache(None)
def _install_PretrainedConfig_patch():
    import transformers

    def _dynamo_overriden_transformers_eq(self, other):
        if not hasattr(other, '__dict__'):
            return False
        return self.__dict__ == other.__dict__
    transformers.configuration_utils.PretrainedConfig.__eq__ = _dynamo_overriden_transformers_eq
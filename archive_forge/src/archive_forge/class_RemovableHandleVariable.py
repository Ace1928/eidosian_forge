import collections
import contextlib
import functools
import importlib
import inspect
import itertools
import random
import sys
import threading
import types
from typing import Dict, List
import torch._dynamo.config
import torch.nn
from torch._guards import TracingContext
from .. import variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, RandomValueSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .ctx_manager import GenericContextWrappingVariable, NullContextVariable
from .dicts import ConstDictVariable
class RemovableHandleVariable(VariableTracker):

    def __init__(self, mutable_local=None, idx=None, **kwargs):
        super().__init__(**kwargs)
        self.mutable_local = mutable_local
        self.idx = idx

    def call_method(self, tx, method_name, args, kwargs):
        if method_name == 'remove':
            tx.output.side_effects.remove_hook(self.idx)
            return variables.ConstantVariable.create(None)
        super().call_method(tx, method_name, args, kwargs)

    def reconstruct(self, codegen):
        if self.user_code_variable_name:
            return [codegen.create_load(self.user_code_variable_name)]
        return super().reconstruct(codegen)
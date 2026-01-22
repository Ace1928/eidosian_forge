import collections
import dataclasses
import functools
import inspect
import itertools
import operator
import sys
import types
from typing import Dict, List
import torch._C
import torch._numpy as tnp
from .. import config, polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, TypeSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
from .user_defined import UserDefinedObjectVariable
class NullVariable(VariableTracker):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return 'NullVariable'

    def reconstruct(self, codegen):
        if sys.version_info < (3, 11):
            unimplemented('cannot reconstruct NullVariable in < Python 3.11')
        return [create_instruction('PUSH_NULL')]
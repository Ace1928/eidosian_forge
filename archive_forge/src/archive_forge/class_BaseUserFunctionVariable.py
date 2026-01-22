import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
class BaseUserFunctionVariable(VariableTracker):

    def get_filename(self):
        return self.get_code().co_filename

    def get_name(self):
        return self.get_code().co_name

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        return tx.inline_user_function_return(self, list(self.self_args()) + list(args), kwargs)

    def inspect_parameter_names(self):
        return list(inspect.signature(self.get_function()).parameters)

    def closure_vars(self, tx):
        return {}
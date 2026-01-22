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
class NumpyVariable(VariableTracker):
    """
    Wrapper around `numpy.*`. Currently, is able to trace a small subset of numpy functions as well as numpy dtypes.
    """

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if not config.trace_numpy:
            unimplemented(f'numpy.{self.value}()')
        from ..utils import numpy_to_tensor_wrapper
        from .tensor import NumpyNdarrayVariable
        if self.value.__name__ == 'dtype':
            unimplemented(f'numpy dtype function is not supported yet. Got type {type(self.value)}.')
        else:
            func = get_np_to_tnp_map().get(self.value)
            if func is None:
                unimplemented(f"Can't find numpy function {self.value} in torch._numpy.  Please file an issue to request support for this function.")
            if func.__module__ == 'torch._numpy.random' and config.use_numpy_random_stream:
                msg = f"delegate '{func.__qualname__}' to NumPy itself via "
                msg += f'confg.use_numpy_random_stream={config.use_numpy_random_stream}'
                unimplemented(msg)
            proxy = tx.output.create_proxy('call_function', numpy_to_tensor_wrapper(func), *proxy_args_kwargs(args, kwargs))
            return NumpyNdarrayVariable.create(tx, proxy)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        unimplemented('numpy')

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def as_proxy(self):
        if config.trace_numpy and isinstance(self.value, type):
            attr = self.value.__name__
            tnp_dtype = tnp.dtype(attr)
            return tnp_dtype.name
        return super().as_proxy()
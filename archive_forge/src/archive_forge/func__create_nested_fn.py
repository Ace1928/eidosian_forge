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
def _create_nested_fn(code, f_globals, name, defaults, closure, kwdefaults, annotations):
    from types import FunctionType
    func = FunctionType(code, f_globals, name, defaults, closure)
    func.__kwdefaults__ = kwdefaults
    if isinstance(annotations, tuple):
        from itertools import pairwise
        annotations = dict(pairwise(annotations))
    assert annotations is None or isinstance(annotations, dict)
    func.__annotations__ = annotations
    return func
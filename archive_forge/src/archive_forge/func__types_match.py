import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
@classmethod
def _types_match(cls, observed, schema_type) -> bool:
    if isinstance(schema_type, torch._C.OptionalType):
        schema_type = schema_type.getElementType()
        return observed is None or cls._types_match(observed, schema_type)
    if isinstance(schema_type, torch._C.AnyType):
        return True
    if schema_type.isSubtypeOf(torch._C.ListType.ofTensors()):
        return isinstance(observed, list) and all((isinstance(i, TensorKey) for i in observed))
    type_map: Tuple[Tuple[Any, Union[type, Tuple[type, ...]]], ...] = ((torch._C.TensorType, TensorKey), (torch._C.NoneType, type(None)), (torch._C.BoolType, bool), (torch._C.IntType, int), (torch._C.FloatType, float), (torch._C.ComplexType, complex), (torch._C.NumberType, (bool, int, float, complex)))
    for jit_type, py_types in type_map:
        if isinstance(schema_type, jit_type):
            return isinstance(observed, py_types)
    return observed is None
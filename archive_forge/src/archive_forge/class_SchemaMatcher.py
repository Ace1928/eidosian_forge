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
class SchemaMatcher:
    """Lookup operator schema based on profiled name.

    When profiling we record the operator's name but not the schema. However
    some analysis requires that information. Fortunately we can look up
    registered schema from the recorded name. We do not, however, record the
    overload and so we must compare the profiled arguments with all overloads
    to determine viable matches.

    Note: Once https://github.com/pytorch/pytorch/issues/78871 is completed
    this code will be obsolete.
    """

    @classmethod
    def inputs_are_mutable(cls, t: _ExtraFields_TorchOp) -> Tuple[Optional[bool], ...]:
        """Determine which inputs may have mutated based on function schema.

        Note that we don't need to resolve down to a single schema to perform
        this analysis. An input is mutable if it is mutable in any overload. In
        practice, however, it is overwhelmingly common to match a single
        overload. If we cannot find any valid schema then we must be
        conservative and assume all inputs are mutable.
        """
        mutable: Optional[List[bool]] = None
        for schema in cls.match_schemas(t):
            mutable = mutable or [False for _ in schema.arguments]
            for i, arg in enumerate(schema.arguments):
                mutable[i] |= getattr(arg.alias_info, 'is_write', False)
        return tuple(mutable or (None for _ in t.inputs))

    @classmethod
    def match_schemas(cls, t: _ExtraFields_TorchOp) -> Tuple[FunctionSchema, ...]:
        signature = tuple((TensorKey.from_tensor(i) if isinstance(i, _TensorMetadata) else [TensorKey.from_tensor(j) for j in i] if isinstance(i, list) else i for i in t.inputs))

        def matches(schema) -> bool:
            return len(schema.arguments) == len(signature) and all((cls._types_match(observed, schema_arg.type) for observed, schema_arg in zip(signature, schema.arguments)))
        return tuple((s for s in cls.lookup_schemas(t.name) or () if matches(s)))

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

    @staticmethod
    def lookup_schemas(name: str) -> Optional[Tuple[FunctionSchema, ...]]:
        try:
            if '::' not in name:
                return None
            return tuple(torch._C._jit_get_schemas_for_operator(name))
        except RuntimeError:
            return None
import operator
from typing import Any, Callable, Dict, Tuple, Optional
import torch
import torch.fx
import torch.fx as fx
from torch.fx import Transformer, Proxy
from torch.fx.node import Argument, Target, Node, map_aggregate
from torch.fx.operator_schemas import (
from .schema_type_annotation import AnnotateTypesWithSchema
class NormalizeOperators(AnnotateTypesWithSchema):
    """
    Normalize callsites that are different ways of "spelling" the same
    invocation into a single, canonical call. Currently supports:

    1. Normalize operators (e.g. operator.add) to the `torch` ops they
       ultimately invoke (e.g. torch.add) when it is possible to statically
       reason that

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = NormalizeOperators(traced).transform()
    """
    binary_magic_method_remap: Dict[Callable[[Any, Any], Any], Callable[[Any, Any], Any]] = {torch.add: operator.add, torch.mul: operator.mul, torch.sub: operator.sub, torch.div: operator.truediv, torch.floor_divide: operator.floordiv, torch.remainder: operator.mod, torch.eq: operator.eq, torch.ne: operator.ne, torch.lt: operator.lt, torch.le: operator.le, torch.gt: operator.gt, torch.ge: operator.ge}

    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
        assert callable(target)
        if target in self.binary_magic_method_remap:
            if len(args) != 2:
                return super().call_function(target, args, kwargs)
            lhs, rhs = args
            return super().call_function(target=self.binary_magic_method_remap[target], args=(lhs, rhs), kwargs={})
        return super().call_function(target, args, kwargs)
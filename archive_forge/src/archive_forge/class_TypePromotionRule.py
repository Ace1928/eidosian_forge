from __future__ import annotations
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set
import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.node import Node  # noqa: F401
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree
class TypePromotionRule(abc.ABC):
    """Base class for type promotion rule per 'torch.ops.{namespace}.{op_name}'."""

    def __init__(self, namespace: str, op_name: str):
        self.namespace = namespace
        self.op_name = op_name

    @abc.abstractmethod
    def __hash__(self) -> int:
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    def is_valid(self) -> bool:
        """Check if the rule is valid."""
        module = getattr(torch.ops, self.namespace)
        py_op = getattr(module, self.op_name, None)
        if py_op is None:
            logger.warning('Cannot find op: %s in module: %s', self.op_name, self.namespace)
            return False
        if not isinstance(py_op, torch._ops.OpOverloadPacket):
            logger.warning('Op: torch.ops.%s.%s is not an OpOverloadPacket, got: %s', self.namespace, self.op_name, type(py_op))
            return False
        return True

    @abc.abstractmethod
    def preview_type_promotion(self, args: tuple, kwargs: dict) -> TypePromotionSnapshot:
        """Preview type promotion results for provided set of args and kwargs.

        Returns a TypePromotionSnapshot object that contains the promoted dtypes for
        the arguments and the expected output dtype.
        """
        ...
import functools
import inspect
import operator
import types
from typing import Dict, List
import sympy
import torch._numpy as tnp
import torch.fx
import torch.random
from torch._dynamo import compiled_autograd
from torch.fx.experimental.symbolic_shapes import (
from .. import config, variables
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from ..utils import (
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import SizeVariable
def unpack_var_sequence(self, tx, idxes=None):
    from .builder import wrap_fx_proxy_cls
    if idxes is None:
        if self.size:
            length = self.size[0]
        else:
            dyn_length = self.call_method(tx, 'size', [ConstantVariable.create(0)], {})
            assert isinstance(dyn_length, (SymNodeVariable, ConstantVariable))
            if isinstance(dyn_length, SymNodeVariable):
                length = dyn_length.evaluate_expr(tx.output)
            else:
                length = dyn_length.value
        idxes = range(length)
    return [wrap_fx_proxy_cls(target_cls=type(self), tx=tx, proxy=self.as_proxy()[i]) for i in idxes]
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
class InsertTypePromotion(_pass.Transform):
    """Explicitly insert type promotion ops to the graph.

    This class subclasses `_pass.Transform` to provide graph level diagnostic tracking.
    Underneath, the main pass is driven by `_TypePromotionInterpreter`, which is a subclass
    of `torch.fx.Interpreter` to interpret the fx.Graph and perform the insertion of type
    promotion operations.

    The interpreter is extended with ability to track diagnostic information for each node.

    By re-running the new and modified nodes using the interpreter, we can update the
    metadata, specifically the fake tensor stored under node.meta["val"], and ensure it
    reflects the latest changes.

    See [FXE0015: fx_node_insert_type_promotion](https://pytorch.org/docs/master/generated/onnx_dynamo_diagnostics_rules/FXE0015%3Afx-node-insert-type-promotion.html) for more details.  # noqa: B950
    """

    def __init__(self, diagnostic_context: diagnostics.DiagnosticContext, module: torch.fx.GraphModule, type_promotion_table: Optional[TypePromotionTable]=None):
        super().__init__(diagnostic_context, module)
        self.interpreter = _TypePromotionInterpreter(diagnostic_context, module, type_promotion_table or TypePromotionTable())

    def _fetch_fake_args(self) -> Sequence[Optional[fake_tensor.FakeTensor]]:
        """Fetch fake args from fx graph.

        For each argument, try to fetch fake tensor from the matching placeholder node.
        """
        fake_args = []
        for node in self.module.graph.nodes:
            if node.op == 'placeholder':
                try:
                    fake_tensor = _fake_tensor_from_node_val(node)
                except RuntimeError as e:
                    if not node.users:
                        fake_tensor = None
                    else:
                        raise RuntimeError('Cannot fetch symbolic fake args from fx graph. InsertTypePromotion pass needs to run with pre-existing fake args, Otherwise the pass will produce inaccurate dynamic shape. ') from e
                fake_args.append(fake_tensor)
        return fake_args

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        assert not args, '`InsertTypePromotion` deduces symbolic fake arguments from the graph. It does not accept concrete arguments as input because this pass requires re-running the graph. When executed with newly faked concrete arguments, the pass loses the symbolic dynamic shape information.'
        assert not kwargs, '`kwargs` is not supported'
        fake_args = self._fetch_fake_args()
        fake_mode = self.fake_mode
        assert fake_mode is not None, 'Cannot detect fake_mode.'
        with proxy_tensor.maybe_disable_fake_tensor_mode(), fake_mode, fx_traceback.preserve_node_meta():
            self.interpreter.run(*fake_args)
        return self.module
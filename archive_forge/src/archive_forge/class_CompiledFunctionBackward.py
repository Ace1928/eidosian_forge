import logging
from contextlib import nullcontext
from functools import wraps
from typing import Any, List, Optional
import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import lazy_format_graph_code
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._logging import getArtifactLogger
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from .. import config
from .dispatch_and_compile_graph import (
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling
from .runtime_wrappers import (
from .schemas import (
from .subclass_utils import unwrap_tensor_subclasses, wrap_tensor_subclasses
from .utils import (
class CompiledFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *unused_args):
        outs = call_compiled_backward()
        if CompiledFunction.maybe_subclass_metadata is not None:
            assert CompiledFunction.maybe_subclass_metadata.grad_input_metas is not None
            outs_wrapped = wrap_tensor_subclasses(outs, subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas)
            return outs_wrapped
        return outs

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError('torch.compile with aot_autograd does not currently support double backward')
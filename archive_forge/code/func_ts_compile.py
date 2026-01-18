import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Callable, Union
import sympy
import torch
from torch import SymInt
import torch.fx as fx
import torch.nn as nn
from torch._decomp import get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols
from .aot_autograd import aot_function, aot_module, make_boxed_compiler
from .compile_utils import strip_overloads
from .partitioners import (
import torch.utils._pytree as pytree
import torch
import torch.fx as fx
from functorch.compile import minifier, check_nvfuser_subprocess, check_nvfuser_correctness_subprocess
from foo import FxModule
@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps) -> Callable:
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """
    with _disable_jit_autocast():
        strip_overloads(fx_g)
        for node in fx_g.graph.nodes:
            if node.target == torch.ops.aten._to_copy and len(node.args) == 1 and (len(node.kwargs) == 1) and ('dtype' in node.kwargs):
                node.target = torch.ops.aten.to
        for node in fx_g.graph.nodes:
            new_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, torch.device):
                    v = v.type
                new_kwargs[k] = v
            node.kwargs = new_kwargs
        fx_g.graph.lint()
        fx_g.recompile()
        f = torch.jit.script(fx_g)
        torch._C._jit_pass_remove_mutation(f.graph)
        f = torch.jit.freeze(f.eval())
        f = torch.jit.optimize_for_inference(f)
        if not any((isinstance(t, torch._subclasses.FakeTensor) for t in inps)):
            f(*inps)
    return f
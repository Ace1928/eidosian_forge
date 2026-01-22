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
class DebugInterpreter(fx.Interpreter):

    def run(self, *args):
        self.symbol_mapping = bind_symbols(self.module, *args)
        super().run(*args)

    def run_node(self, n):

        def subst_symint(ni):
            if not isinstance(ni, SymInt):
                return ni
            r = sympy.expand(ni.node.expr.xreplace(self.symbol_mapping))
            assert r.is_number, r
            return int(r)

        def subst_symint_tuple(nis):
            return tuple((subst_symint(ni) for ni in nis))

        def check_significant_strides(a, b):
            if subst_symint(a.numel()) > 0:
                for idx in range(a.ndim):
                    if subst_symint(a.stride(idx)) != b.stride(idx) and subst_symint(a.size(idx)) > 1:
                        return False
            return True

        def check(nv, rv, desc):
            assert callable(desc)
            assert nv.dtype == rv.dtype, f'{desc()}: {nv.dtype} != {rv.dtype}'
            assert subst_symint_tuple(nv.size()) == rv.size(), f'{desc()}: {nv.size()} aka {subst_symint_tuple(nv.size())} != {rv.size()}'
            same_strides = check_significant_strides(nv, rv)
            assert same_strides, f'{desc()}: {nv.stride()} aka {subst_symint_tuple(nv.stride())} != {rv.stride()}'
        r = super().run_node(n)
        if 'val' in n.meta:
            n_vals, n_spec = pytree.tree_flatten(n.meta['val'])
            r_vals, r_spec = pytree.tree_flatten(r)
            assert len(n_vals) == len(r_vals), f'{len(n_vals)} != {len(r_vals)}'
            for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
                if not isinstance(rv, torch.Tensor):
                    continue
                check(nv, rv, lambda: f'output {i} where {self.symbol_mapping}')
        return r
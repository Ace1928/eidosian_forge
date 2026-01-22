import builtins
import functools
import inspect
import itertools
import logging
import sys
import textwrap
import time
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import patch
import sympy
import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity, preserve_rng_state
from . import config, ir
from .autotune_process import TensorMeta, TritonBenchmarkRequest
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import ChoiceCaller, IndentedBuffer, KernelTemplate
from .codegen.triton import texpr, TritonKernel, TritonPrinter, TritonScheduling
from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .utils import do_bench, Placeholder, sympy_dot, sympy_product, unique
from .virtualized import V
from . import lowering  # noqa: F401
class ExternKernelChoice:

    def __init__(self, kernel, cpp_kernel=None, *, name=None, has_out_variant=True):
        super().__init__()
        name = name or kernel.__name__
        assert callable(kernel)
        assert not hasattr(extern_kernels, name), 'duplicate extern kernel'
        self.name = name
        self.cpp_kernel = cpp_kernel
        self.has_out_variant = has_out_variant
        setattr(extern_kernels, name, kernel)

    def to_callable(self):
        return getattr(extern_kernels, self.name)

    def call_name(self):
        return f'extern_kernels.{self.name}'

    @functools.lru_cache(None)
    def hash_key(self):
        fn = self.to_callable()
        parts = [self.name, getattr(fn, '__name__', ''), getattr(fn, '__module__', '')]
        try:
            parts.append(inspect.getsource(fn))
        except Exception:
            pass
        return code_hash('-'.join(parts))

    def bind(self, input_nodes, layout, ordered_kwargs_for_cpp_kernel=(), **kwargs):
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        return ExternKernelCaller(self, input_nodes, layout, kwargs, has_out_variant=self.has_out_variant)
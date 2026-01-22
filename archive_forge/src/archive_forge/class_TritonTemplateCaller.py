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
class TritonTemplateCaller(ChoiceCaller):

    def __init__(self, name, input_nodes, layout, make_kernel_render, debug_extra, bmreq):
        super().__init__(name, input_nodes, layout)
        self.make_kernel_render = make_kernel_render
        self.debug_extra = debug_extra
        self.bmreq = bmreq

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
        return f'TritonTemplateCaller({self.bmreq.module_path}, {self.debug_extra})'

    def call_name(self):
        return f'template_kernels.{self.name}'

    def hash_key(self):
        return '-'.join([self.name.rsplit('_', 1)[0], self.bmreq.module_cache_key])

    def output_node(self):
        return ir.TensorBox.create(ir.TritonTemplateBuffer(layout=self.layout, inputs=self.input_nodes, make_kernel_render=self.make_kernel_render))
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
class KernelGroup:

    def __init__(self):
        super().__init__()
        self.args = KernelArgs()
        self.loops_code = BracesBuffer()
        self.ws = WorkSharing(self.loops_code)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(self.ws)
        self.scheduled_nodes = []

    def new_kernel(self, cls, *args):
        return cls(self.args, parallel_num_threads(), *args)

    def finalize_kernel(self, new_kernel, nodes):
        self.scheduled_nodes += nodes
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)

    def get_num_args(self):
        arg_defs, call_args, arg_types = self.args.cpp_argdefs()
        args_num = len(arg_defs)
        return args_num

    def codegen_define_and_call(self, wrapper):
        self.stack.close()
        if not self.scheduled_nodes:
            return
        fused_name = get_fused_kernel_name(self.scheduled_nodes, config.cpp.descriptive_names) if config.cpp.descriptive_names else ''
        kernel_name = '_'.join(['cpp', fused_name, wrapper.next_kernel_suffix()])
        arg_defs, call_args, arg_types = self.args.cpp_argdefs()
        arg_defs = ',\n'.ljust(25).join(arg_defs)
        arg_types = ','.join(arg_types)
        code = BracesBuffer()
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform == 'linux'
        if enable_kernel_profile:
            code.writelines(['#include <ATen/record_function.h>'])
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else 'kernel'
        code.writeline(codecache.cpp_prefix())
        code.writeline(f'extern "C" void {kernel_decl_name}({arg_defs})')
        with code.indent():
            if enable_kernel_profile:
                graph_id = V.graph.graph_id
                prefix = 'graph_' + str(graph_id) + '_' if graph_id is not None else ''
                code.writelines([f'RECORD_FUNCTION("{prefix + kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'])
            for old, new in self.args.aliases():
                code.writeline(f'auto {old} = {new};')
            code.splice(self.loops_code)
        codecache_def = IndentedBuffer()
        if not V.graph.cpp_wrapper:
            codecache_def.writeline("async_compile.cpp('''")
        codecache_def.splice(code)
        if not V.graph.cpp_wrapper:
            codecache_def.writeline("''')")
        codecache_str = codecache_def.getvalue()
        codecache_str = codecache_str.replace('#pragma CMT', '//')
        wrapper.define_kernel(kernel_name, codecache_str, cuda=False)
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)
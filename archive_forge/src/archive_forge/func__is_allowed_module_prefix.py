import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
def _is_allowed_module_prefix(obj):
    allowed_modules = ('torch', 'math')
    disallowed_modules = ['torch.optim', 'torch.nn.modules.rnn', 'torch._dynamo', 'torch._C._dynamo', 'torch._inductor', 'torch._C.inductor', 'torch.fx', 'torch._C._autograd', 'torch._C._cudart', 'torch._C._distributed_autograd', 'torch._C._distributed_c10d', 'torch._C._distributed_rpc', 'torch._C._functorch', 'torch._C._monitor', 'torch._C._nvtx', 'torch._C._lazy', 'torch._C._profiler', 'torch.__config__', 'torch._custom_op', 'torch._dispatch', 'torch._export', 'torch._functorch.make_functional', 'torch._functorch.compile_utils', 'torch._functorch.partitioners', 'torch._functorch.aot_autograd', 'torch._functorch.compilers', 'torch._functorch.fx_minifier', 'torch.autograd.profiler_util', 'torch.autograd.profiler', 'torch._jit_internal', 'torch._library', 'torch._lobpcg', 'torch._logging', 'torch._meta_registrations', 'torch._namedtensor_internals', 'torch._numpy', 'torch._sources', 'torch._subclasses', 'torch._tensor', 'torch._tensor_str', 'torch._utils', 'torch._utils_internal', 'torch._vmap_internals', 'torch.compiler', 'torch.distributed', 'torch.export', 'torch.hub', 'torch.jit', 'torch.library', 'torch.masked.maskedtensor', 'torch.nn.init', 'torch.nn.modules.module', 'torch.nn.parallel', 'torch.nn.utils', 'torch.multiprocessing', 'torch.onnx', 'torch.overrides', 'torch.package', 'torch.profiler', 'torch.serialization', 'torch.storage', 'torch.utils']
    if config.trace_distributed:
        disallowed_modules.append('torch.distributed.')
    allowed_modules_dot = tuple([x + '.' for x in allowed_modules])
    module = inspect.getmodule(obj)
    if module is None:
        return False
    mod_name = module.__name__
    if any((mod_name.startswith(m) for m in disallowed_modules)):
        return False
    return mod_name in allowed_modules or mod_name.startswith(allowed_modules_dot)
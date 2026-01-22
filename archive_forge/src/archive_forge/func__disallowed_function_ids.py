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
@FunctionIdSet
def _disallowed_function_ids() -> Set[int]:
    remove: List[Any] = [True, False, None, collections.OrderedDict, copy.copy, copy.deepcopy, inspect.signature, math.__package__, torch.__builtins__, torch.autocast_decrement_nesting, torch.autocast_increment_nesting, torch.autograd.grad, torch.clear_autocast_cache, torch.cuda.current_device, torch.cuda.set_device, torch.distributions.constraints.is_dependent, torch.distributions.normal.Normal, torch.inference_mode, torch.jit.isinstance, torch.set_anomaly_enabled, torch.set_autocast_cache_enabled, torch.set_autocast_cpu_dtype, torch.set_autocast_cpu_enabled, torch.set_autocast_enabled, torch.set_autocast_gpu_dtype, warnings.warn, torch._C._dynamo.eval_frame.unsupported, torch.Tensor.__init__, torch.resize_as_, torch._tensor._convert]
    dtypes = [obj for obj in torch.__dict__.values() if isinstance(obj, type(torch.float32))]
    remove += dtypes
    storage = [obj for obj in torch.__dict__.values() if isinstance(obj, type(torch.FloatStorage))]
    remove += storage
    if torch.distributed.is_available():
        remove.extend(torch.distributed.distributed_c10d.dynamo_unsupported_distributed_c10d_ops)
    return {id(x) for x in remove}
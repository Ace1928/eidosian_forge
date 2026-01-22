import contextlib
import dataclasses
import functools
import logging
import os
import sys
import time
import warnings
from itertools import count
from typing import (
from unittest import mock
from functorch.compile import min_cut_rematerialization_partition
import torch._functorch.config as functorch_config
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo import (
from torch._dynamo.utils import detect_fake_mode, lazy_format_graph_code
from torch._functorch.aot_autograd import aot_export_module, make_boxed_func
from torch._inductor.codecache import code_hash, CompiledFxGraph, FxGraphCache
from torch._inductor.debug import save_args_for_compile_fx_inner
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from .._dynamo.backends.common import aot_autograd
from ..fx.graph import _PyTreeCodeGen
from . import config, metrics
from .debug import DebugContext
from .decomposition import select_decomp_table
from .fx_passes.joint_graph import joint_graph_passes
from .fx_passes.post_grad import post_grad_passes, view_to_reshape
from .fx_passes.pre_grad import pre_grad_passes
from .graph import GraphLowering
from .ir import ExternKernelNode
from .utils import get_dtype_size, has_incompatible_cudagraph_ops
from .virtualized import V
def _unlift_graph(mod, gm, graph_signature):
    state_dict = {}
    for name, param in mod.named_parameters(remove_duplicate=False):
        state_dict[name] = param
    for name, param in mod.named_buffers(remove_duplicate=False):
        state_dict[name] = param
    from torch._export.exported_program import _construct_inp_pos_to_param_buffer_name, _unlift
    inp_pos_to_param_buffer_name = _construct_inp_pos_to_param_buffer_name(gm, graph_signature, state_dict, {})
    unlifted_gm = _unlift(gm, inp_pos_to_param_buffer_name, pytree.LeafSpec(), None, state_dict, {}, graph_signature.buffers_to_mutate)
    return unlifted_gm
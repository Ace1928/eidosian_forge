import itertools
from contextlib import nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import dynamo_timed, preserve_rng_state
from torch._guards import detect_fake_mode
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
from . import config
from .partitioners import default_partition
from ._aot_autograd.utils import (  # noqa: F401
from ._aot_autograd.logging_utils import (  # noqa: F401
from ._aot_autograd.functional_utils import (  # noqa: F401
from ._aot_autograd.schemas import (  # noqa: F401
from ._aot_autograd.subclass_utils import (  # noqa: F401
from ._aot_autograd.collect_metadata_analysis import (  # noqa: F401
from ._aot_autograd.input_output_analysis import (  # noqa: F401
from ._aot_autograd.traced_function_transforms import (  # noqa: F401
from ._aot_autograd.runtime_wrappers import (  # noqa: F401
from ._aot_autograd.dispatch_and_compile_graph import (  # noqa: F401
from ._aot_autograd.jit_compile_runtime_wrappers import (  # noqa: F401
def aot_module_simplified(mod: nn.Module, args, fw_compiler: Callable, bw_compiler: Optional[Callable]=None, partition_fn: Callable=default_partition, decompositions: Optional[Dict]=None, keep_inference_input_mutations=False, inference_compiler: Optional[Callable]=None) -> nn.Module:
    """
    This is the simplified or low overhead version of aot_module. For frontends
    like TorchDynamo, the input functions/modules to AOT are static and have
    unpacked inputs/outputs. This gives us an opportunity to remove the
        (1) pytree overhead to parse inputs/outputs,
        (2) AOT Autograd cache,
        (3) Reading of params/buffers in every forward call

    :func:`aot_module_simplified` removes these overheads.
    """
    params = {**dict(mod.named_parameters(remove_duplicate=False)), **dict(mod.named_buffers(remove_duplicate=False))}
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)
    params_len = len(params_flat)
    functional_call = create_functional_call(mod, params_spec, params_len)
    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler
    seen_sources = set()
    full_args = []
    full_args.extend(params_flat)
    if (tracing_context := torch._guards.TracingContext.try_get()):
        tracing_context.params_flat = params_flat
    aot_autograd_arg_pos_to_source = None
    if hasattr(mod, '_param_name_to_source'):
        aot_autograd_arg_pos_to_source = []
        for name in params.keys():
            assert name in mod._param_name_to_source, f'{name} not found.'
            source = mod._param_name_to_source[name]
            assert source not in seen_sources, source
            seen_sources.add(source)
            aot_autograd_arg_pos_to_source.append(source)
    full_args.extend(args)
    if hasattr(mod, 'graph'):
        for i, node in enumerate(mod.graph.nodes):
            if node.op == 'placeholder':
                if hasattr(node, '_dynamo_source'):
                    if aot_autograd_arg_pos_to_source is None:
                        aot_autograd_arg_pos_to_source = []
                    source = node._dynamo_source
                    assert source not in seen_sources, source
                    seen_sources.add(source)
                    aot_autograd_arg_pos_to_source.append(source)
    if aot_autograd_arg_pos_to_source is not None:
        assert len(full_args) == len(aot_autograd_arg_pos_to_source)
    dynamic_shapes = False
    for x in full_args:
        if isinstance(x, FakeTensor):
            dynamic_shapes = x.fake_mode.shape_env is not None
            break
    aot_config = AOTConfig(fw_compiler=fw_compiler, bw_compiler=bw_compiler, inference_compiler=inference_compiler, partition_fn=partition_fn, decompositions=decompositions, num_params_buffers=params_len, aot_id=next(AOT_COUNTER), keep_inference_input_mutations=keep_inference_input_mutations, dynamic_shapes=dynamic_shapes, aot_autograd_arg_pos_to_source=aot_autograd_arg_pos_to_source, is_export=False, no_tangents=False)
    with compiled_autograd.disable():
        compiled_fn = create_aot_dispatcher_function(functional_call, full_args, aot_config)

    def forward(*runtime_args):
        full_args = []
        full_args.extend(params_flat)
        full_args.extend(runtime_args)
        return compiled_fn(full_args)
    forward.zero_grad = mod.zero_grad
    forward.named_parameters = mod.named_parameters
    forward.named_buffers = mod.named_buffers
    return forward
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
def _aot_export_function(func: Callable, args, *, num_params_buffers: int=0, decompositions: Optional[Dict]=None, no_tangents: bool=False) -> Tuple[torch.fx.GraphModule, ViewAndMutationMeta, pytree.TreeSpec, pytree.TreeSpec]:
    dynamic_shapes = False
    for x in args:
        if isinstance(x, FakeTensor):
            dynamic_shapes = x.fake_mode.shape_env is not None
            break
    flat_fn, out_spec = create_tree_flattened_fn(func, args)
    flat_args, in_spec = pytree.tree_flatten(args)
    aot_config = AOTConfig(fw_compiler=None, bw_compiler=None, inference_compiler=None, partition_fn=None, decompositions=decompositions, num_params_buffers=num_params_buffers, aot_id=next(AOT_COUNTER), keep_inference_input_mutations=False, dynamic_shapes=dynamic_shapes, aot_autograd_arg_pos_to_source=None, is_export=True, no_tangents=no_tangents)
    fx_g, meta = create_aot_dispatcher_function(flat_fn, flat_args, aot_config)
    return (fx_g, meta, in_spec, out_spec.spec)
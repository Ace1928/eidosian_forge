import threading
from typing import Any, Dict
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(*, kernel_idx, grid, kwargs):
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code
    kernel = kernel_side_table.get_kernel(kernel_idx)
    if len(grid) == 1:
        grid_fn = grid[0]
    else:
        fn_name, code = user_defined_kernel_grid_fn_code(kernel.fn.__name__, kernel.configs, grid)
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        grid_fn = namespace[fn_name]
    kernel[grid_fn](**kwargs)
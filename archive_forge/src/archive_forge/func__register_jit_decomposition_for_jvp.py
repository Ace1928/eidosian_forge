import inspect
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch._decomp
from torch import Tensor
from torch._prims_common.wrappers import _maybe_remove_out_wrapper
def _register_jit_decomposition_for_jvp(decomp, use_python=False):
    if decomp in decomposition_table_for_jvp:
        decomposition_table_used = decomposition_table_for_jvp
    elif decomp in decomposition_table:
        decomposition_table_used = decomposition_table
    else:
        raise RuntimeError(f'could not find decomposition for {decomp}')
    decomp_fn = decomposition_table_used[decomp]
    decomp_fn = _maybe_remove_out_wrapper(decomp_fn)
    if use_python:
        decomp_fn = torch.jit.ignore(decomp_fn)
        sig = inspect.signature(decomp_fn)

        def get_function_def(sig):
            param_def = [f'{param_str}' for param_str in sig.parameters.values()]
            param_use = [f'{param_str}' for param_str in sig.parameters.keys()]
            return f'def wrapped_decomp({', '.join(param_def)}):\n  return decomp_fn({', '.join(param_use)})\n'
        f_str = get_function_def(sig)
        graph = torch.jit.CompilationUnit(f_str).wrapped_decomp.graph
    else:
        graph = torch.jit.script(decomp_fn).graph
    torch.jit._register_decomposition(decomp, graph)
from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module
def _get_inflate_helper_fn_name(arg_idx: int, input_idx: int, function_name: str) -> str:
    return f'_inflate_helper_for_{function_name}_input_{input_idx}_arg_{arg_idx}'
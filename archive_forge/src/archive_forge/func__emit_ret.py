import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _emit_ret(ret):
    return _emit_type(ret.type)
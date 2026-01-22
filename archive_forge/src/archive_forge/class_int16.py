import builtins
import torch
from . import _dtypes_impl
class int16(signedinteger):
    name = 'int16'
    typecode = 'h'
    torch_dtype = torch.int16
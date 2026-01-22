import builtins
import torch
from . import _dtypes_impl
class int8(signedinteger):
    name = 'int8'
    typecode = 'b'
    torch_dtype = torch.int8
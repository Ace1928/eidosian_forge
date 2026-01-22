import builtins
import torch
from . import _dtypes_impl
class complex128(complexfloating):
    name = 'complex128'
    typecode = 'D'
    torch_dtype = torch.complex128
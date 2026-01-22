import builtins
import torch
from . import _dtypes_impl
class complex64(complexfloating):
    name = 'complex64'
    typecode = 'F'
    torch_dtype = torch.complex64
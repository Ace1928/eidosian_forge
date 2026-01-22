import copy
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings
import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer
class Int8Params(torch.nn.Parameter):

    def __new__(cls, data=None, requires_grad=True, has_fp16_weights=False, CB=None, SCB=None):
        cls.has_fp16_weights = has_fp16_weights
        cls.CB = None
        cls.SCB = None
        if data is None:
            data = torch.empty(0)
        obj = torch.Tensor._make_subclass(cls, data, requires_grad)
        obj.CB, obj.SCB = (cls.CB, cls.SCB)
        return obj

    def cuda(self, device):
        if self.has_fp16_weights:
            return super().cuda(device)
        else:
            B = self.data.contiguous().half().cuda(device)
            CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
            del CBt
            del SCBt
            self.data = CB
            self.CB = CB
            self.SCB = SCB
        return self

    @overload
    def to(self: T, device: Optional[Union[int, device]]=..., dtype: Optional[Union[dtype, str]]=..., non_blocking: bool=...) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool=...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool=...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == 'cuda' and (self.data.device.type == 'cpu'):
            return self.cuda(device)
        else:
            new_param = Int8Params(super().to(device=device, dtype=dtype, non_blocking=non_blocking), requires_grad=self.requires_grad, has_fp16_weights=self.has_fp16_weights)
            new_param.CB = self.CB
            new_param.SCB = self.SCB
            return new_param
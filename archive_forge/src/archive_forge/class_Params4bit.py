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
class Params4bit(torch.nn.Parameter):

    def __new__(cls, data: Optional[torch.Tensor]=None, requires_grad=False, quant_state: Optional[QuantState]=None, blocksize: int=64, compress_statistics: bool=True, quant_type: str='fp4', quant_storage: torch.dtype=torch.uint8, module: Optional['Linear4bit']=None, bnb_quantized: bool=False) -> 'Params4bit':
        if data is None:
            data = torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self.data = data
        self.module = module
        return self

    def __getstate__(self):
        state = self.__dict__
        state['data'] = self.data
        state['requires_grad'] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.requires_grad = state['requires_grad']
        self.blocksize = state['blocksize']
        self.compress_statistics = state['compress_statistics']
        self.quant_type = state['quant_type']
        self.quant_state = state['quant_state']
        self.data = state['data']
        self.quant_storage = state['quant_storage']
        self.bnb_quantized = state['bnb_quantized']
        self.module = state['module']

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quant_state = copy.deepcopy(state['quant_state'])
        new_instance.data = copy.deepcopy(state['data'])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    @classmethod
    def from_prequantized(cls, data: torch.Tensor, quantized_stats: Dict[str, Any], requires_grad: bool=False, device='cuda', **kwargs) -> 'Params4bit':
        self = torch.Tensor._make_subclass(cls, data.to(device))
        self.requires_grad = requires_grad
        self.quant_state = QuantState.from_dict(qs_dict=quantized_stats, device=device)
        self.blocksize = self.quant_state.blocksize
        self.compress_statistics = self.quant_state.nested
        self.quant_type = self.quant_state.quant_type
        self.bnb_quantized = True
        return self

    def _quantize(self, device):
        w = self.data.contiguous().cuda(device)
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type, quant_storage=self.quant_storage)
        self.data = w_4bit
        self.quant_state = quant_state
        if self.module is not None:
            self.module.quant_state = quant_state
        self.bnb_quantized = True
        return self

    def cuda(self, device: Optional[Union[int, device, str]]=None, non_blocking: bool=False):
        return self.to(device='cuda' if device is None else device, non_blocking=non_blocking)

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
        if device is not None and device.type == 'cuda' and (not self.bnb_quantized):
            return self._quantize(device)
        else:
            if self.quant_state is not None:
                self.quant_state.to(device)
            new_param = Params4bit(super().to(device=device, dtype=dtype, non_blocking=non_blocking), requires_grad=self.requires_grad, quant_state=self.quant_state, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type)
            return new_param
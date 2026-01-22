import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import (
from torch.overrides import (
from torch.utils.dlpack import DLDeviceType
def _reduce_ex_internal(self, proto):
    check_serializing_named_tensor(self)
    torch.utils.hooks.warn_if_has_hooks(self)
    backward_hooks: Dict[Any, Any] = OrderedDict()
    if self.device.type in ['xla', 'mtia', 'ort'] or (not torch._C._has_storage(self) and self.device.type == torch._C._get_privateuse1_backend_name()):
        numpy_tensor = self.cpu().numpy() if self.dtype != torch.bfloat16 else self.cpu().to(torch.float32).numpy()
        return (torch._utils._rebuild_device_tensor_from_numpy, (numpy_tensor, self.dtype, str(self.device), self.requires_grad))
    if self.device.type == 'meta':
        arg_meta = (self.dtype, tuple(self.size()), self.stride(), self.requires_grad)
        return (torch._utils._rebuild_meta_tensor_no_storage, arg_meta)
    if self.is_quantized:
        quantizer_params: Union[Tuple[torch.qscheme, float, int], Tuple[Any, Tensor, Tensor, int]]
        if self.qscheme() == torch.per_tensor_affine:
            quantizer_params = (torch.per_tensor_affine, self.q_scale(), self.q_zero_point())
        elif self.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
            quantizer_params = (torch.per_channel_affine, self.q_per_channel_scales(), self.q_per_channel_zero_points(), self.q_per_channel_axis())
        else:
            raise RuntimeError(f'Serialization is not supported for tensors of type {self.qscheme()}')
        args_qtensor = (torch.storage.TypedStorage(wrap_storage=self._typed_storage()._untyped_storage, dtype=self.dtype, _internal=True), self.storage_offset(), tuple(self.size()), self.stride(), quantizer_params, self.requires_grad, backward_hooks)
        return (torch._utils._rebuild_qtensor, args_qtensor)
    elif self.is_sparse:
        if self.layout == torch.sparse_coo:
            args_sparse = (self.layout, (self._indices(), self._values(), self.size(), self.is_coalesced()))
        else:
            raise NotImplementedError(f'sparse tensor __reduce_ex__ for layout `{self.layout}`')
        return (torch._utils._rebuild_sparse_tensor, args_sparse)
    elif self.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
        if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices, plain_indices = (self.crow_indices(), self.col_indices())
        else:
            compressed_indices, plain_indices = (self.ccol_indices(), self.row_indices())
        args_sparse_compressed = (self.layout, (compressed_indices, plain_indices, self.values(), self.size()))
        return (torch._utils._rebuild_sparse_tensor, args_sparse_compressed)
    elif self.is_nested:
        args_nested = (self.values(), self._nested_tensor_size(), self._nested_tensor_strides(), self._nested_tensor_storage_offsets())
        return (torch._utils._rebuild_nested_tensor, args_nested)
    elif self.data_ptr() == 0 and type(self) is not torch.Tensor and (type(self).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__):
        arg_wrapper_subclass = (type(self), self.dtype, tuple(self.size()), self.stride(), self.storage_offset(), self.layout, self.device, self.requires_grad)
        return (torch._utils._rebuild_wrapper_subclass, arg_wrapper_subclass)
    else:
        v3_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn, torch.bits8, torch.bits16, torch.bits1x8, torch.bits2x4, torch.bits4x2]
        if self.dtype in v3_dtypes:
            rebuild_func = torch._utils._rebuild_tensor_v3
            storage = self.untyped_storage()
        else:
            rebuild_func = torch._utils._rebuild_tensor_v2
            storage = torch.storage.TypedStorage(wrap_storage=self._typed_storage()._untyped_storage, dtype=self.dtype, _internal=True)
        args = (storage, self.storage_offset(), tuple(self.size()), self.stride(), self.requires_grad, backward_hooks)
        if isinstance(storage, torch.storage.UntypedStorage):
            args = args + (self.dtype,)
        metadata = torch._utils.get_tensor_metadata(self)
        if metadata:
            args = args + (metadata,)
        return (rebuild_func, args)
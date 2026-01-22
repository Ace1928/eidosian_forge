import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
class _StorageBase:
    _cdata: Any
    is_sparse: bool = False
    is_sparse_csr: bool = False
    device: torch.device

    def __init__(self, *args, **kwargs):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx):
        ...

    def __setitem__(self, *args, **kwargs):
        ...

    def copy_(self, source: T, non_blocking: _Optional[bool]=None) -> T:
        ...

    def new(self) -> T:
        ...

    def nbytes(self) -> int:
        ...

    def size(self) -> int:
        return self.nbytes()

    def type(self, dtype: _Optional[str]=None, non_blocking: bool=False) -> T:
        ...

    def cuda(self, device=None, non_blocking=False, **kwargs) -> T:
        ...

    def hpu(self, device=None, non_blocking=False, **kwargs) -> T:
        ...

    def element_size(self) -> int:
        ...

    def get_device(self) -> int:
        return self.device.index

    def data_ptr(self) -> int:
        ...

    def _share_filename_cpu_(self, *args, **kwargs):
        ...

    def _share_fd_cpu_(self, *args, **kwargs):
        ...

    @classmethod
    def _new_using_filename_cpu(cls: Type[T], size: int) -> T:
        ...

    @classmethod
    def _new_using_fd_cpu(cls: Type[T], size: int) -> T:
        ...

    @classmethod
    def from_buffer(cls: Type[T], *args, **kwargs) -> T:
        ...

    @classmethod
    def _new_shared_filename_cpu(cls: Type[T], manager, obj, size, *, device=None, dtype=None) -> T:
        ...

    @classmethod
    def _release_ipc_counter_cuda(cls: Type[T], *args, **kwargs) -> T:
        ...

    @classmethod
    def _new_with_weak_ptr(cls: Type[T], *args, **kwargs) -> T:
        ...

    def _shared_decref(self) -> T:
        ...

    def _write_file(self, *args, **kwargs):
        ...

    def resize_(self, size: int):
        ...

    def _weak_ref(self, *args, **kwargs) -> T:
        ...

    def _set_from_file(self, *args, **kwargs):
        ...

    def _set_cdata(self, *args, **kwargs):
        ...

    def _share_cuda_(self, *args, **kwargs):
        ...

    def is_shared(self) -> bool:
        ...

    @classmethod
    def _new_shared_cuda(cls: Type[T], *args, **kwargs) -> T:
        ...

    def _shared_incref(self, *args, **kwargs):
        ...

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        ...

    @property
    def is_cuda(self):
        ...

    @property
    def is_hpu(self):
        ...

    @classmethod
    def from_file(cls, filename, shared, nbytes) -> T:
        ...

    @classmethod
    def _expired(cls, *args, **kwargs) -> T:
        ...

    def _byteswap(self, *args, **kwargs):
        ...

    def _get_filename(self, *args, **kwargs) -> _Optional[str]:
        ...

    def __str__(self):
        info_str = f'[{torch.typename(self)}(device={self.device}) of size {len(self)}]'
        if self.device.type == 'meta':
            return '...\n' + info_str
        else:
            data_str = ' ' + '\n '.join((str(self[i]) for i in range(self.size())))
            return data_str + '\n' + info_str

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter((self[i] for i in range(self.size())))

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def __sizeof__(self):
        return super().__sizeof__() + self.size()

    def clone(self):
        """Return a copy of this storage."""
        return type(self)(self.nbytes(), device=self.device).copy_(self)

    def tolist(self):
        """Return a list containing the elements of this storage."""
        return list(self)

    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
        if self.device.type != 'cpu':
            return torch.UntypedStorage(self.size()).copy_(self, False)
        else:
            return self

    def mps(self):
        """Return a MPS copy of this storage if it's not already on the MPS."""
        if self.device.type != 'mps':
            return torch.UntypedStorage(self.size(), device='mps').copy_(self, False)
        else:
            return self

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = torch.tensor([], dtype=torch.uint8, device=self.device).set_(cast(Storage, self)).to(dtype)._typed_storage()
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        """Casts this storage to double type."""
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type."""
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type."""
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type."""
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type."""
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type."""
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type."""
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type."""
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type."""
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type."""
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type."""
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
        return self._to(torch.float8_e4m3fn)

    def is_pinned(self, device: Union[str, torch.device]='cuda'):
        """Determine whether the CPU storage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'cuda'``.

        Returns:
            A boolean variable.
        """
        return torch.tensor([], dtype=torch.uint8, device=self.device).set_(cast(Storage, self)).is_pinned(device)

    def pin_memory(self, device: Union[str, torch.device]='cuda'):
        """Copy the CPU storage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'cuda'``.

        Returns:
            A pinned CPU storage.
        """
        if self.device.type != 'cpu':
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")
        pinned_tensor = torch.tensor([], dtype=torch.uint8, device=self.device).set_(cast(Storage, self)).pin_memory(device)
        return pinned_tensor.untyped_storage()

    def share_memory_(self):
        """See :meth:`torch.UntypedStorage.share_memory_`"""
        from torch.multiprocessing import get_sharing_strategy
        if self.device.type in ['cuda', torch._C._get_privateuse1_backend_name()]:
            pass
        elif get_sharing_strategy() == 'file_system':
            self._share_filename_cpu_()
        else:
            self._share_fd_cpu_()
        return self

    @classmethod
    def _new_shared(cls, size, *, device='cpu'):
        """Create a new storage in shared memory with the same data type."""
        from torch.multiprocessing import get_sharing_strategy
        device = torch.device(device)
        if device.type in ['cuda', torch._C._get_privateuse1_backend_name()]:
            return cls(size, device=device)
        elif get_sharing_strategy() == 'file_system':
            return cls._new_using_filename_cpu(size)
        else:
            return cls._new_using_fd_cpu(size)

    def untyped(self):
        return self

    def byteswap(self, dtype):
        """Swap bytes in underlying data."""
        elem_size = torch._utils._element_size(dtype)
        if dtype.is_complex:
            elem_size = max(int(elem_size / 2), 1)
        self._byteswap(elem_size)
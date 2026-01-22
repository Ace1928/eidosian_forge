from __future__ import annotations
from typing import (
from pandas.core.interchange.dataframe_protocol import (
class PandasBuffer(Buffer):
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(self, x: np.ndarray, allow_copy: bool=True) -> None:
        """
        Handle only regular columns (= numpy arrays) for now.
        """
        if x.strides[0] and (not x.strides == (x.dtype.itemsize,)):
            if allow_copy:
                x = x.copy()
            else:
                raise RuntimeError('Exports cannot be zero-copy in the case of a non-contiguous buffer')
        self._x = x

    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
        return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
        return self._x.__array_interface__['data'][0]

    def __dlpack__(self) -> Any:
        """
        Represent this structure as DLPack interface.
        """
        return self._x.__dlpack__()

    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        """
        return (DlpackDeviceType.CPU, None)

    def __repr__(self) -> str:
        return 'PandasBuffer(' + str({'bufsize': self.bufsize, 'ptr': self.ptr, 'device': self.__dlpack_device__()[0].name}) + ')'
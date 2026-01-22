from __future__ import annotations
from typing import (
from pandas.core.interchange.dataframe_protocol import (
class PandasBufferPyarrow(Buffer):
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(self, buffer: pa.Buffer, *, length: int) -> None:
        """
        Handle pyarrow chunked arrays.
        """
        self._buffer = buffer
        self._length = length

    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
        return self._buffer.size

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
        return self._buffer.address

    def __dlpack__(self) -> Any:
        """
        Represent this structure as DLPack interface.
        """
        raise NotImplementedError()

    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        """
        return (DlpackDeviceType.CPU, None)

    def __repr__(self) -> str:
        return 'PandasBuffer[pyarrow](' + str({'bufsize': self.bufsize, 'ptr': self.ptr, 'device': 'CPU'}) + ')'
import enum
from typing import Tuple
import numpy as np
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.utils import _inherit_docstrings
@_inherit_docstrings(ProtocolBuffer)
class PandasProtocolBuffer(ProtocolBuffer):
    """
    Data in the buffer is guaranteed to be contiguous in memory.

    Note that there is no dtype attribute present, a buffer can be thought of
    as simply a block of memory. However, if the column that the buffer is
    attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
    implemented, then that dtype information will be contained in the return
    value from ``__dlpack__``.

    This distinction is useful to support both (a) data exchange via DLPack on a
    buffer and (b) dtypes like variable-length strings which do not have a
    fixed number of bytes per element.

    Parameters
    ----------
    x : np.ndarray
        Data to be held by ``Buffer``.
    allow_copy : bool, default: True
        A keyword that defines whether or not the library is allowed
        to make a copy of the data. For example, copying data would be necessary
        if a library supports strided buffers, given that this protocol
        specifies contiguous buffers. Currently, if the flag is set to ``False``
        and a copy is needed, a ``RuntimeError`` will be raised.
    """

    def __init__(self, x: np.ndarray, allow_copy: bool=True) -> None:
        if not x.strides == (x.dtype.itemsize,):
            if allow_copy:
                x = x.copy()
            else:
                raise RuntimeError('Exports cannot be zero-copy in the case ' + 'of a non-contiguous buffer')
        self._x = x

    @property
    def bufsize(self) -> int:
        return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        return self._x.__array_interface__['data'][0]

    def __dlpack__(self):
        raise NotImplementedError('__dlpack__')

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:

        class Device(enum.IntEnum):
            CPU = 1
        return (Device.CPU, None)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular ``PandasProtocolBuffer``.

        Returns
        -------
        str
        """
        return 'Buffer(' + str({'bufsize': self.bufsize, 'ptr': self.ptr, 'device': self.__dlpack_device__()[0].name}) + ')'
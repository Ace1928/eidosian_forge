from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, TypedDict
from .utils import ColumnNullType, DlpackDeviceType, DTypeKind
class ProtocolBuffer(ABC):
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
    """

    @property
    @abstractmethod
    def bufsize(self) -> int:
        """
        Buffer size in bytes.

        Returns
        -------
        int
        """
        pass

    @property
    @abstractmethod
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def __dlpack__(self) -> Any:
        """
        Produce DLPack capsule (see array API standard).

        DLPack not implemented in NumPy yet, so leave it out here.

        Raises
        ------
        ``TypeError`` if the buffer contains unsupported dtypes.
        ``NotImplementedError`` if DLPack support is not implemented.

        Notes
        -----
        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        """
        pass

    @abstractmethod
    def __dlpack_device__(self) -> Tuple[DlpackDeviceType, Optional[int]]:
        """
        Device type and device ID for where the data in the buffer resides.

        Uses device type codes matching DLPack. Enum members are:
            - CPU = 1
            - CUDA = 2
            - CPU_PINNED = 3
            - OPENCL = 4
            - VULKAN = 7
            - METAL = 8
            - VPI = 9
            - ROCM = 10

        Returns
        -------
        tuple
            Device type and device ID.

        Notes
        -----
        Must be implemented even if ``__dlpack__`` is not.
        """
        pass
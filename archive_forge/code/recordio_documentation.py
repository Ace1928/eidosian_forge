from collections import namedtuple
from multiprocessing import current_process
import ctypes
import struct
import numbers
import numpy as np
from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
Inserts input record at given index.

        Examples
        ---------
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        idx : int
            Index of a file.
        buf :
            Record to write.
        
from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
def wait_to_read(self):
    """Waits until all previous write operations on the current array are finished.

        This method guarantees that all previous write operations that pushed
        into the backend engine for execution are actually finished.

        Examples
        --------
        >>> import time
        >>> tic = time.time()
        >>> a = mx.nd.ones((1000,1000))
        >>> b = mx.nd.dot(a, a)
        >>> print(time.time() - tic) # doctest: +SKIP
        0.003854036331176758
        >>> b.wait_to_read()
        >>> print(time.time() - tic) # doctest: +SKIP
        0.0893700122833252
        """
    check_call(_LIB.MXNDArrayWaitToRead(self.handle))
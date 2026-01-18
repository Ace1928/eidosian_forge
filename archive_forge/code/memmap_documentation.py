from contextlib import nullcontext
import numpy as np
from .._utils import set_module
from .numeric import uint8, ndarray, dtype
from numpy.compat import os_fspath, is_pathlib_path

        Write any changes in the array to the file on disk.

        For further information, see `memmap`.

        Parameters
        ----------
        None

        See Also
        --------
        memmap

        
import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def _fread3(fobj):
    """Read a 3-byte int from an open binary file object

    Parameters
    ----------
    fobj : file
        File descriptor

    Returns
    -------
    n : int
        A 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, '>u1', 3).astype(np.int64)
    return (b1 << 16) + (b2 << 8) + b3
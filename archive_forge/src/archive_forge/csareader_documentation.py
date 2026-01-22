import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
Strip string to first null

    Parameters
    ----------
    s : bytes

    Returns
    -------
    sdash : str
       s stripped to first occurrence of null (0)
    
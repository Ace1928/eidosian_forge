import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def typecode(self):
    """
        Return the typecode of the variable.

        Returns
        -------
        typecode : char
            The character typecode of the variable (e.g., 'i' for int).

        """
    return self._typecode
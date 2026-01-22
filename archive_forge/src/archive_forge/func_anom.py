import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def anom(self, axis=None, dtype=None):
    """
        Compute the anomalies (deviations from the arithmetic mean)
        along the given axis.

        Returns an array of anomalies, with the same shape as the input and
        where the arithmetic mean is computed along the given axis.

        Parameters
        ----------
        axis : int, optional
            Axis over which the anomalies are taken.
            The default is to use the mean of the flattened array as reference.
        dtype : dtype, optional
            Type to use in computing the variance. For arrays of integer type
             the default is float32; for arrays of float types it is the same as
             the array type.

        See Also
        --------
        mean : Compute the mean of the array.

        Examples
        --------
        >>> a = np.ma.array([1,2,3])
        >>> a.anom()
        masked_array(data=[-1.,  0.,  1.],
                     mask=False,
               fill_value=1e+20)

        """
    m = self.mean(axis, dtype)
    if not axis:
        return self - m
    else:
        return self - expand_dims(m, axis)
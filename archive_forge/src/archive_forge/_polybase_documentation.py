import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
Convert series to series of this class.

        The `series` is expected to be an instance of some polynomial
        series of one of the types supported by by the numpy.polynomial
        module, but could be some other class that supports the convert
        method.

        .. versionadded:: 1.7.0

        Parameters
        ----------
        series : series
            The series instance to be converted.
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.

        Returns
        -------
        new_series : series
            A series of the same kind as the calling class and equal to
            `series` when evaluated.

        See Also
        --------
        convert : similar instance method

        
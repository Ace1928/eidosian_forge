from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
def uarray(nominal_values, std_devs=None):
    """
    Return a NumPy array of numbers with uncertainties
    initialized with the given nominal values and standard
    deviations.

    nominal_values, std_devs -- valid arguments for numpy.array, with
    identical shapes (list of numbers, list of lists, numpy.ndarray,
    etc.).

    std_devs=None is only used for supporting legacy code, where
    nominal_values can be the tuple of nominal values and standard
    deviations.
    """
    if std_devs is None:
        deprecation('uarray() should now be called with two arguments.')
        nominal_values, std_devs = nominal_values
    return numpy.vectorize(lambda v, s: uncert_core.Variable(v, s), otypes=[object])(nominal_values, std_devs)
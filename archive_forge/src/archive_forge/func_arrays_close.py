from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
import copy
import weakref
import math
from math import isnan, isinf
import random
import sys
import uncertainties.core as uncert_core
from uncertainties.core import ufloat, AffineScalarFunc, ufloat_fromstr
from uncertainties import umath
def arrays_close(m1, m2, precision=0.0001):
    """
        Returns True iff m1 and m2 are almost equal, where elements
        can be either floats or AffineScalarFunc objects.

        Two independent AffineScalarFunc objects are deemed equal if
        both their nominal value and uncertainty are equal (up to the
        given precision).

        m1, m2 -- NumPy arrays.

        precision -- precision passed through to
        uncertainties.test_uncertainties.numbers_close().
        """
    for elmt1, elmt2 in zip(m1.flat, m2.flat):
        elmt1 = uncert_core.to_affine_scalar(elmt1)
        elmt2 = uncert_core.to_affine_scalar(elmt2)
        if not numbers_close(elmt1.nominal_value, elmt2.nominal_value, precision):
            return False
        if not numbers_close(elmt1.std_dev, elmt2.std_dev, precision):
            return False
    return True
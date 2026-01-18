from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
def ufloat_fromstr(representation, tag=None):
    """
    Return a new random variable (Variable object) from a string.

    Strings 'representation' of the form '12.345+/-0.015',
    '12.345(15)', '12.3' or u'1.2±0.1' (Unicode string) are recognized
    (see more complete list below).  In the last case, an uncertainty
    of +/-1 is assigned to the last digit.

    Invalid representations raise a ValueError.

    This function tries to parse back most of the formats that are made
    available by this module. Examples of valid string representations:

        12.3e10+/-5e3
        (-3.1415 +/- 0.0001)e+02  # Factored exponent

        # Pretty-print notation (only with a unicode string):
        12.3e10 ± 5e3  # ± symbol
        (12.3 ± 5.0) × 10⁻¹²  # Times symbol, superscript
        12.3 ± 5e3  # Mixed notation (± symbol, but e exponent)

        # Double-exponent values:
        (-3.1415 +/- 1e-4)e+200
        (1e-20 +/- 3)e100

        0.29
        31.
        -31.
        31
        -3.1e10

        -1.23(3.4)
        -1.34(5)
        1(6)
        3(4.2)
        -9(2)
        1234567(1.2)
        12.345(15)
        -12.3456(78)e-6
        12.3(0.4)e-5
        169.0(7)
        169.1(15)
        .123(4)
        .1(.4)

        # NaN uncertainties:
        12.3(nan)
        12.3(NAN)
        3±nan

    Surrounding spaces are ignored.

    About the "shorthand" notation: 1.23(3) = 1.23 ± 0.03 but
    1.23(3.) = 1.23 ± 3.00. Thus, the presence of a decimal point in
    the uncertainty signals an absolute uncertainty (instead of an
    uncertainty on the last digits of the nominal value).
    """
    nominal_value, std_dev = str_to_number_with_uncert(representation.strip())
    return ufloat(nominal_value, std_dev, tag)
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
def test_value_construction():
    """
    Tests the various means of constructing a constant number with
    uncertainty *without a string* (see test_ufloat_fromstr(), for this).
    """
    x = ufloat(3, 0.14)
    assert x.nominal_value == 3
    assert x.std_dev == 0.14
    assert x.tag is None
    x = ufloat(3, 0.14, 'pi')
    assert x.nominal_value == 3
    assert x.std_dev == 0.14
    assert x.tag == 'pi'
    x = ufloat(3, 0.14, tag='pi')
    assert x.nominal_value == 3
    assert x.std_dev == 0.14
    assert x.tag == 'pi'
    representation = (3, 0.14)
    x = ufloat(3, 0.14)
    x2 = ufloat(representation)
    assert x.nominal_value == x2.nominal_value
    assert x.std_dev == x2.std_dev
    assert x.tag is None
    assert x2.tag is None
    x = ufloat(3, 0.14, 'pi')
    x2 = ufloat(representation, 'pi')
    assert x.nominal_value == x2.nominal_value
    assert x.std_dev == x2.std_dev
    assert x.tag == 'pi'
    assert x2.tag == 'pi'
    x = ufloat(3, 0.14, tag='pi')
    x2 = ufloat(representation, tag='pi')
    assert x.nominal_value == x2.nominal_value
    assert x.std_dev == x2.std_dev
    assert x.tag == 'pi'
    assert x2.tag == 'pi'
    try:
        x = ufloat(3, -0.1)
    except uncert_core.NegativeStdDev:
        pass
    try:
        x = ufloat((3, -0.1))
    except uncert_core.NegativeStdDev:
        pass
    try:
        ufloat(1)
    except:
        pass
    else:
        raise Exception('An exception should be raised')
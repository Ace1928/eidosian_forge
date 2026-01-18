import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_safe_scalar_isnan():
    assert not safe_scalar_isnan(True)
    assert not safe_scalar_isnan(None)
    assert not safe_scalar_isnan('sadf')
    assert not safe_scalar_isnan((1, 2, 3))
    assert not safe_scalar_isnan(np.asarray([1, 2, 3]))
    assert not safe_scalar_isnan([np.nan])
    assert safe_scalar_isnan(np.nan)
    assert safe_scalar_isnan(np.float32(np.nan))
    assert safe_scalar_isnan(float(np.nan))
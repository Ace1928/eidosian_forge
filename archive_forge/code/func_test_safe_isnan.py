import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_safe_isnan():
    assert np.array_equal(safe_isnan([1, True, None, np.nan, 'asdf']), [False, False, False, True, False])
    assert safe_isnan(np.nan).ndim == 0
    assert safe_isnan(np.nan)
    assert not safe_isnan(None)
    assert not safe_isnan('asdf')
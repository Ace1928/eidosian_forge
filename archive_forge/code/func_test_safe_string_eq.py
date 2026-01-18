import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_safe_string_eq():
    assert safe_string_eq('foo', 'foo')
    assert not safe_string_eq('foo', 'bar')
    if not six.PY3:
        assert safe_string_eq(unicode('foo'), 'foo')
    assert not safe_string_eq(np.empty((2, 2)), 'foo')
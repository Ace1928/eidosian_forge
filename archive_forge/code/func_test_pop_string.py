from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_pop_string():
    f = PlainTextFormatter()
    type_str = '%s.%s' % (C.__module__, 'C')
    with pytest.raises(KeyError):
        f.pop(type_str)
    f.for_type(type_str, foo_printer)
    f.pop(type_str)
    with pytest.raises(KeyError):
        f.lookup_by_type(C)
    with pytest.raises(KeyError):
        f.pop(type_str)
    f.for_type(C, foo_printer)
    assert f.pop(type_str, None) is foo_printer
    with pytest.raises(KeyError):
        f.lookup_by_type(C)
    with pytest.raises(KeyError):
        f.pop(type_str)
    assert f.pop(type_str, None) is None
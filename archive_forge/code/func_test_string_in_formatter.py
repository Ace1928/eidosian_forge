from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_string_in_formatter():
    f = PlainTextFormatter()
    type_str = '%s.%s' % (C.__module__, 'C')
    f.for_type(type_str, foo_printer)
    assert type_str in f
    assert C in f
from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_warn_error_for_type():
    f = HTMLFormatter()
    f.for_type(int, lambda i: name_error)
    with capture_output() as captured:
        result = f(5)
    assert result is None
    assert 'Traceback' in captured.stdout
    assert 'NameError' in captured.stdout
    assert 'name_error' in captured.stdout
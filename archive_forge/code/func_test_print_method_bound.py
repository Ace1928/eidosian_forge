from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_print_method_bound():
    f = HTMLFormatter()

    class MyHTML(object):

        def _repr_html_(self):
            return 'hello'
    with capture_output() as captured:
        result = f(MyHTML)
    assert result is None
    assert 'FormatterWarning' not in captured.stderr
    with capture_output() as captured:
        result = f(MyHTML())
    assert result == 'hello'
    assert captured.stderr == ''
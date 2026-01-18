from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_print_method_weird():

    class TextMagicHat(object):

        def __getattr__(self, key):
            return key
    f = HTMLFormatter()
    text_hat = TextMagicHat()
    assert text_hat._repr_html_ == '_repr_html_'
    with capture_output() as captured:
        result = f(text_hat)
    assert result is None
    assert 'FormatterWarning' not in captured.stderr

    class CallableMagicHat(object):

        def __getattr__(self, key):
            return lambda: key
    call_hat = CallableMagicHat()
    with capture_output() as captured:
        result = f(call_hat)
    assert result is None

    class BadReprArgs(object):

        def _repr_html_(self, extra, args):
            return 'html'
    bad = BadReprArgs()
    with capture_output() as captured:
        result = f(bad)
    assert result is None
    assert 'FormatterWarning' not in captured.stderr
from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_pretty_max_seq_length():
    f = PlainTextFormatter(max_seq_length=1)
    lis = list(range(3))
    text = f(lis)
    assert text == '[0, ...]'
    f.max_seq_length = 0
    text = f(lis)
    assert text == '[0, 1, 2]'
    text = f(list(range(1024)))
    lines = text.splitlines()
    assert len(lines) == 1024
import sys
from IPython.utils.PyColorize import Parser
import io
import pytest
def test_parse_sample(style):
    """and test writing to a buffer"""
    buf = io.StringIO()
    p = Parser(style=style)
    p.format(sample, buf)
    buf.seek(0)
    f1 = buf.read()
    assert 'ERROR' not in f1
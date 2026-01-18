import sys
import pytest
from IPython.utils import capture
@pytest.mark.parametrize('method_mime', _mime_map.items())
def test_rich_output_empty(method_mime):
    """RichOutput with no args"""
    rich = capture.RichOutput()
    method, mime = method_mime
    assert getattr(rich, method)() is None
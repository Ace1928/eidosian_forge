import sys
import pytest
from IPython.utils import capture
@pytest.mark.parametrize('method_mime', _mime_map.items())
def test_rich_output_no_metadata(method_mime):
    """test RichOutput with no metadata"""
    data = full_data
    rich = capture.RichOutput(data=data)
    method, mime = method_mime
    assert getattr(rich, method)() == data[mime]
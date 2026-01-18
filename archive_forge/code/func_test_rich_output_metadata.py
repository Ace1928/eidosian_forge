import sys
import pytest
from IPython.utils import capture
@pytest.mark.parametrize('method_mime', _mime_map.items())
def test_rich_output_metadata(method_mime):
    """test RichOutput with metadata"""
    data = full_data
    metadata = full_metadata
    rich = capture.RichOutput(data=data, metadata=metadata)
    method, mime = method_mime
    assert getattr(rich, method)() == (data[mime], metadata[mime])
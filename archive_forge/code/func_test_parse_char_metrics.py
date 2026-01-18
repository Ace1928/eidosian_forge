from io import BytesIO
import pytest
import logging
from matplotlib import _afm
from matplotlib import font_manager as fm
def test_parse_char_metrics():
    fh = BytesIO(AFM_TEST_DATA)
    _afm._parse_header(fh)
    metrics = _afm._parse_char_metrics(fh)
    assert metrics == ({0: (250.0, 'space', [0, 0, 0, 0]), 42: (1141.0, 'foo', [40, 60, 800, 360]), 99: (583.0, 'bar', [40, -10, 543, 210])}, {'space': (250.0, 'space', [0, 0, 0, 0]), 'foo': (1141.0, 'foo', [40, 60, 800, 360]), 'bar': (583.0, 'bar', [40, -10, 543, 210])})
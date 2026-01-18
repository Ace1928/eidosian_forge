from io import BytesIO
import pytest
import logging
from matplotlib import _afm
from matplotlib import font_manager as fm
def test_parse_header():
    fh = BytesIO(AFM_TEST_DATA)
    header = _afm._parse_header(fh)
    assert header == {b'StartFontMetrics': 2.0, b'FontName': 'MyFont-Bold', b'EncodingScheme': 'FontSpecific', b'FullName': 'My Font Bold', b'FamilyName': 'Test Fonts', b'Weight': 'Bold', b'ItalicAngle': 0.0, b'IsFixedPitch': False, b'UnderlinePosition': -100, b'UnderlineThickness': 56.789, b'Version': '001.000', b'Notice': b'Copyright \xa9 2017 No one.', b'FontBBox': [0, -321, 1234, 369], b'StartCharMetrics': 3}
import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_python_specific_encodings_not_used_in_xml_declaration(self):
    markup = b'<?xml version="1.0"?>\n<foo/>'
    soup = self.soup(markup)
    for encoding in PYTHON_SPECIFIC_ENCODINGS:
        if encoding in ('idna', 'mbcs', 'oem', 'undefined', 'string_escape', 'string-escape'):
            continue
        encoded = soup.encode(encoding)
        assert b'<?xml version="1.0"?>' in encoded
        assert encoding.encode('ascii') not in encoded
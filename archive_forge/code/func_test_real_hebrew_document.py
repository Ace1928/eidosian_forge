import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_real_hebrew_document(self):
    hebrew_document = b'<html><head><title>Hebrew (ISO 8859-8) in Visual Directionality</title></head><body><h1>Hebrew (ISO 8859-8) in Visual Directionality</h1>\xed\xe5\xec\xf9</body></html>'
    soup = self.soup(hebrew_document, from_encoding='iso8859-8')
    assert soup.original_encoding in ('iso8859-8', 'iso-8859-8')
    assert soup.encode('utf-8') == hebrew_document.decode('iso8859-8').encode('utf-8')
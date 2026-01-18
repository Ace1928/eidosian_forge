import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_prettify_accepts_formatter_function(self):
    soup = BeautifulSoup('<html><body>foo</body></html>', 'html.parser')
    pretty = soup.prettify(formatter=lambda x: x.upper())
    assert 'FOO' in pretty
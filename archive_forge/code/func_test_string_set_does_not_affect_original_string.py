from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_string_set_does_not_affect_original_string(self):
    soup = self.soup('<a><b>foo</b><c>bar</c>')
    soup.b.string = soup.c.string
    assert soup.a.encode() == b'<a><b>bar</b><c>bar</c></a>'
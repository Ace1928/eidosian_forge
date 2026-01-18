from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_with_maintains_next_element_throughout(self):
    soup = self.soup('<p><a>one</a><b>three</b></p>')
    a = soup.a
    b = a.contents[0]
    a.insert(1, 'two')
    left, right = a.contents
    left.replaceWith('')
    right.replaceWith('')
    assert 'three' == soup.b.string
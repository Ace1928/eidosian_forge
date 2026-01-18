import warnings
from bs4.element import (
from . import SoupTest
def test_tag_with_recursive_string_has_string(self):
    soup = self.soup('<a><b>foo</b></a>')
    assert soup.a.string == 'foo'
    assert soup.string == 'foo'
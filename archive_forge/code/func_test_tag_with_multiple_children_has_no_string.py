import warnings
from bs4.element import (
from . import SoupTest
def test_tag_with_multiple_children_has_no_string(self):
    soup = self.soup('<a>foo<b></b><b></b></b>')
    assert soup.b.string == None
    soup = self.soup('<a>foo<b></b>bar</b>')
    assert soup.b.string == None
    soup = self.soup('<a>foo</b>')
    soup.a.insert(1, 'bar')
    assert soup.a.string == None
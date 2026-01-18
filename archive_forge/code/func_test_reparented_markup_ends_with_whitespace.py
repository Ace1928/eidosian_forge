import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
def test_reparented_markup_ends_with_whitespace(self):
    markup = '<p><em>foo</p>\n<p>bar<a></a></em></p>\n'
    soup = self.soup(markup)
    assert '<body><p><em>foo</em></p><em>\n</em><p><em>bar<a></a></em></p>\n</body>' == soup.body.decode()
    assert 2 == len(soup.find_all('p'))
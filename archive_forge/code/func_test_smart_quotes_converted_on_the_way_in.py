import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_smart_quotes_converted_on_the_way_in(self):
    quote = b'<p>\x91Foo\x92</p>'
    soup = self.soup(quote)
    assert soup.p.string == '‘Foo’'
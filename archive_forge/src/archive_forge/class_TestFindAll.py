from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestFindAll(SoupTest):
    """Basic tests of the find_all() method."""

    def test_find_all_text_nodes(self):
        """You can search the tree for text nodes."""
        soup = self.soup('<html>Foo<b>bar</b>»</html>')
        assert soup.find_all(string='bar') == ['bar']
        assert soup.find_all(string=['Foo', 'bar']) == ['Foo', 'bar']
        assert soup.find_all(string=re.compile('.*')) == ['Foo', 'bar', '»']
        assert soup.find_all(string=True) == ['Foo', 'bar', '»']

    def test_find_all_limit(self):
        """You can limit the number of items returned by find_all."""
        soup = self.soup('<a>1</a><a>2</a><a>3</a><a>4</a><a>5</a>')
        self.assert_selects(soup.find_all('a', limit=3), ['1', '2', '3'])
        self.assert_selects(soup.find_all('a', limit=1), ['1'])
        self.assert_selects(soup.find_all('a', limit=10), ['1', '2', '3', '4', '5'])
        self.assert_selects(soup.find_all('a', limit=0), ['1', '2', '3', '4', '5'])

    def test_calling_a_tag_is_calling_findall(self):
        soup = self.soup("<a>1</a><b>2<a id='foo'>3</a></b>")
        self.assert_selects(soup('a', limit=1), ['1'])
        self.assert_selects(soup.b(id='foo'), ['3'])

    def test_find_all_with_self_referential_data_structure_does_not_cause_infinite_recursion(self):
        soup = self.soup('<a></a>')
        l = []
        l.append(l)
        assert [] == soup.find_all(l)

    def test_find_all_resultset(self):
        """All find_all calls return a ResultSet"""
        soup = self.soup('<a></a>')
        result = soup.find_all('a')
        assert hasattr(result, 'source')
        result = soup.find_all(True)
        assert hasattr(result, 'source')
        result = soup.find_all(string='foo')
        assert hasattr(result, 'source')
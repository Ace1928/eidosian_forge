from pdb import set_trace
import pickle
import pytest
import warnings
from bs4.builder import (
from bs4.builder._htmlparser import BeautifulSoupHTMLParser
from . import SoupTest, HTMLTreeBuilderSmokeTest
def test_on_duplicate_attribute(self):
    markup = '<a class="cls" href="url1" href="url2" href="url3" id="id">'
    soup = self.soup(markup)
    assert 'url3' == soup.a['href']
    assert ['cls'] == soup.a['class']
    assert 'id' == soup.a['id']

    def assert_attribute(on_duplicate_attribute, expected):
        soup = self.soup(markup, on_duplicate_attribute=on_duplicate_attribute)
        assert expected == soup.a['href']
        assert ['cls'] == soup.a['class']
        assert 'id' == soup.a['id']
    assert_attribute(None, 'url3')
    assert_attribute(BeautifulSoupHTMLParser.REPLACE, 'url3')
    assert_attribute(BeautifulSoupHTMLParser.IGNORE, 'url1')

    def accumulate(attrs, key, value):
        if not isinstance(attrs[key], list):
            attrs[key] = [attrs[key]]
        attrs[key].append(value)
    assert_attribute(accumulate, ['url1', 'url2', 'url3'])
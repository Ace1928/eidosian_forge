import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
class TestEncoding(SoupTest):
    """Test the ability to encode objects into strings."""

    def test_unicode_string_can_be_encoded(self):
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert soup.b.string.encode('utf-8') == '☃'.encode('utf-8')

    def test_tag_containing_unicode_string_can_be_encoded(self):
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert soup.b.encode('utf-8') == html.encode('utf-8')

    def test_encoding_substitutes_unrecognized_characters_by_default(self):
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert soup.b.encode('ascii') == b'<b>&#9731;</b>'

    def test_encoding_can_be_made_strict(self):
        html = '<b>☃</b>'
        soup = self.soup(html)
        with pytest.raises(UnicodeEncodeError):
            soup.encode('ascii', errors='strict')

    def test_decode_contents(self):
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert '☃' == soup.b.decode_contents()

    def test_encode_contents(self):
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert '☃'.encode('utf8') == soup.b.encode_contents(encoding='utf8')

    def test_encode_deeply_nested_document(self):
        limit = sys.getrecursionlimit() + 1
        markup = '<span>' * limit
        soup = self.soup(markup)
        encoded = soup.encode()
        assert limit == encoded.count(b'<span>')

    def test_deprecated_renderContents(self):
        html = '<b>☃</b>'
        soup = self.soup(html)
        soup.renderContents()
        assert '☃'.encode('utf8') == soup.b.renderContents()

    def test_repr(self):
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert html == repr(soup)
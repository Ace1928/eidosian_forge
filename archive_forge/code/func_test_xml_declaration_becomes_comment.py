import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_xml_declaration_becomes_comment(self):
    markup = '<?xml version="1.0" encoding="utf-8"?><html></html>'
    soup = self.soup(markup)
    assert isinstance(soup.contents[0], Comment)
    assert soup.contents[0] == '?xml version="1.0" encoding="utf-8"?'
    assert 'html' == soup.contents[0].next_element.name
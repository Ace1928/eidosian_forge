import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_tag_containing_unicode_string_can_be_encoded(self):
    html = '<b>☃</b>'
    soup = self.soup(html)
    assert soup.b.encode('utf-8') == html.encode('utf-8')
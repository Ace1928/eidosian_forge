import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_popping_namespaced_tag(self):
    markup = '<rss xmlns:dc="foo"><dc:creator>b</dc:creator><dc:date>2012-07-02T20:33:42Z</dc:date><dc:rights>c</dc:rights><image>d</image></rss>'
    soup = self.soup(markup)
    assert str(soup.rss) == markup
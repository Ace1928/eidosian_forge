import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_unicode_pickle(self):
    html = '<b>☃</b>'
    soup = self.soup(html)
    dumped = pickle.dumps(soup, pickle.HIGHEST_PROTOCOL)
    loaded = pickle.loads(dumped)
    assert loaded.decode() == soup.decode()
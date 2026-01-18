import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_prettify_can_encode_data(self):
    soup = self.soup('<a></a>')
    assert bytes == type(soup.prettify('utf-8'))
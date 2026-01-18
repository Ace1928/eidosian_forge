import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_tag_with_no_attributes_can_have_attributes_added(self):
    data = self.soup('<a>text</a>')
    data.a['foo'] = 'bar'
    assert '<a foo="bar">text</a>' == data.a.decode()
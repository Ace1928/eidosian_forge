from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_unwrap_returns_thing_that_was_replaced(self):
    text = '<a><b></b><c></c></a>'
    soup = self.soup(text)
    a = soup.a
    new_a = a.unwrap()
    assert a == new_a
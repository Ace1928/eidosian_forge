from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_first_child(self):
    data = '<a><b></b><c></c></a>'
    soup = self.soup(data)
    soup.b.replace_with(soup.c)
    assert '<a><c></c></a>' == soup.decode()
from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_last_child(self):
    data = '<a><b></b><c></c></a>'
    soup = self.soup(data)
    soup.c.replace_with(soup.b)
    assert '<a><b></b></a>' == soup.decode()
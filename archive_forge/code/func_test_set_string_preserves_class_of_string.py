from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_set_string_preserves_class_of_string(self):
    soup = self.soup('<a></a>')
    cdata = CData('foo')
    soup.a.string = cdata
    assert isinstance(soup.a.string, CData)
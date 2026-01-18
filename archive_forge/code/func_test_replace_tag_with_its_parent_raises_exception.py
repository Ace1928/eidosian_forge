from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_tag_with_its_parent_raises_exception(self):
    text = '<a><b></b></a>'
    soup = self.soup(text)
    with pytest.raises(ValueError):
        soup.b.replace_with(soup.a)
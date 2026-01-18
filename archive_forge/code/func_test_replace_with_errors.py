from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_with_errors(self):
    a_tag = Tag(name='a')
    with pytest.raises(ValueError):
        a_tag.replace_with("won't work")
    a_tag = self.soup('<a><b></b></a>').a
    with pytest.raises(ValueError):
        a_tag.b.replace_with(a_tag)
    with pytest.raises(ValueError):
        a_tag.b.replace_with('string1', a_tag, 'string2')
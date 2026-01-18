from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_parent_of_text_element(self):
    text = self.tree.find(string='Start here')
    assert text.parent.name == 'b'
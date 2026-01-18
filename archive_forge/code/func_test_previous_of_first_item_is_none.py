from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_previous_of_first_item_is_none(self):
    first = self.tree.find('html')
    assert first.previous_element == None
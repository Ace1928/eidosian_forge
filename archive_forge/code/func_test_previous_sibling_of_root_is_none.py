from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_previous_sibling_of_root_is_none(self):
    assert self.tree.previous_sibling == None
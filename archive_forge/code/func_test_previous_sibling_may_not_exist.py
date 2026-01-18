from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_previous_sibling_may_not_exist(self):
    assert self.tree.html.previous_sibling == None
    nested_span = self.tree.find(id='1.1')
    assert nested_span.previous_sibling == None
    first_span = self.tree.find(id='1')
    assert first_span.previous_sibling == None
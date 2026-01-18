from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_parent_generator(self):
    parents = [parent['id'] for parent in self.start.parents if parent is not None and 'id' in parent.attrs]
    assert parents, ['bottom', 'middle' == 'top']
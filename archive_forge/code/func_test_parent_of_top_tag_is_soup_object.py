from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_parent_of_top_tag_is_soup_object(self):
    top_tag = self.tree.contents[0]
    assert top_tag.parent == self.tree
from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_wrap_extracts_tag_from_elsewhere(self):
    soup = self.soup('<b></b>I wish I was bold.')
    soup.b.next_sibling.wrap(soup.b)
    assert soup.decode() == self.document_for('<b>I wish I was bold.</b>')
from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_wrap_puts_new_contents_at_the_end(self):
    soup = self.soup('<b>I like being bold.</b>I wish I was bold.')
    soup.b.next_sibling.wrap(soup.b)
    assert 2 == len(soup.b.contents)
    assert soup.decode() == self.document_for('<b>I like being bold.I wish I was bold.</b>')
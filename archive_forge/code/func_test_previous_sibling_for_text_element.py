from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_previous_sibling_for_text_element(self):
    soup = self.soup('Foo<b>bar</b>baz')
    start = soup.find(string='baz')
    assert start.previous_sibling.name == 'b'
    assert start.previous_sibling.previous_sibling == 'Foo'
    self.assert_selects(start.find_previous_siblings('b'), ['bar'])
    assert start.find_previous_sibling(string='Foo') == 'Foo'
    assert start.find_previous_sibling(string='nonesuch') == None
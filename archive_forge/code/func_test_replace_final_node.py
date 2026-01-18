from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_final_node(self):
    soup = self.soup('<b>Argh!</b>')
    soup.find(string='Argh!').replace_with('Hooray!')
    new_text = soup.find(string='Hooray!')
    b = soup.b
    assert new_text.previous_element == b
    assert new_text.parent == b
    assert new_text.previous_element.next_element == new_text
    assert new_text.next_element == None
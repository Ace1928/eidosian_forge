from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_replace_with(self):
    soup = self.soup("<p>There's <b>no</b> business like <b>show</b> business</p>")
    no, show = soup.find_all('b')
    show.replace_with(no)
    assert soup.decode() == self.document_for("<p>There's  business like <b>no</b> business</p>")
    assert show.parent == None
    assert no.parent == soup.p
    assert no.next_element == 'no'
    assert no.next_sibling == ' business'
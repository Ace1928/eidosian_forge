import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
def test_reparented_markup_containing_children(self):
    markup = '<div><a>aftermath<p><noscript>target</noscript>aftermath</a></p></div>'
    soup = self.soup(markup)
    noscript = soup.noscript
    assert 'target' == noscript.next_element
    target = soup.find(string='target')
    final_aftermath = soup.find_all(string='aftermath')[-1]
    assert final_aftermath == target.next_element
    assert target == final_aftermath.previous_element
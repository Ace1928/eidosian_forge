import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
def test_reparented_markup_containing_identical_whitespace_nodes(self):
    """Verify that we keep the two whitespace nodes in this
        document distinct when reparenting the adjacent <tbody> tags.
        """
    markup = '<table> <tbody><tbody><ims></tbody> </table>'
    soup = self.soup(markup)
    space1, space2 = soup.find_all(string=' ')
    tbody1, tbody2 = soup.find_all('tbody')
    assert space1.next_element is tbody1
    assert tbody2.next_element is space2
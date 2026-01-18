import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_p_tag_is_never_empty_element(self):
    """A <p> tag is never designated as an empty-element tag.

        Even if the markup shows it as an empty-element tag, it
        shouldn't be presented that way.
        """
    soup = self.soup('<p/>')
    assert not soup.p.is_empty_element
    assert str(soup.p) == '<p></p>'
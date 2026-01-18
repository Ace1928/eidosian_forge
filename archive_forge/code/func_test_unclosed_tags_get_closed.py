import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_unclosed_tags_get_closed(self):
    """A tag that's not closed by the end of the document should be closed.

        This applies to all tags except empty-element tags.
        """
    self.assert_soup('<p>', '<p></p>')
    self.assert_soup('<b>', '<b></b>')
    self.assert_soup('<br>', '<br/>')
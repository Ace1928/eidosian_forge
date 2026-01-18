import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_quotes_not_html_substituted(self):
    """There's no need to do this except inside attribute values."""
    text = 'Bob\'s "bar"'
    assert self.sub.substitute_html(text) == text
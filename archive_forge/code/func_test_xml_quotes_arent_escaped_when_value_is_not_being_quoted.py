import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_xml_quotes_arent_escaped_when_value_is_not_being_quoted(self):
    quoted = 'Welcome to "Bob\'s Bar"'
    assert self.sub.substitute_xml(quoted) == quoted
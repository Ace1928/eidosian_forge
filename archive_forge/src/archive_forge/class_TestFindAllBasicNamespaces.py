from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestFindAllBasicNamespaces(SoupTest):

    def test_find_by_namespaced_name(self):
        soup = self.soup('<mathml:msqrt>4</mathml:msqrt><a svg:fill="red">')
        assert '4' == soup.find('mathml:msqrt').string
        assert 'a' == soup.find(attrs={'svg:fill': 'red'}).name
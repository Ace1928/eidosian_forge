from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
def test_tag_inherits_self_closing_rules_from_builder(self):
    html_soup = BeautifulSoup('', 'html.parser')
    html_br = html_soup.new_tag('br')
    html_p = html_soup.new_tag('p')
    assert b'<br/>' == html_br.encode()
    assert b'<p></p>' == html_p.encode()
from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_previous_generator(self):
    start = self.tree.find(string='One')
    predecessors = [node for node in start.previous_elements]
    b, body, head, html = predecessors
    assert b['id'] == '1'
    assert body.name == 'body'
    assert head.name == 'head'
    assert html.name == 'html'
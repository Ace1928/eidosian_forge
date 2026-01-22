from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestPreviousOperations(ProximityTest):

    def setup_method(self):
        super(TestPreviousOperations, self).setup_method()
        self.end = self.tree.find(string='Three')

    def test_previous(self):
        assert self.end.previous_element['id'] == '3'
        assert self.end.previous_element.previous_element == 'Two'

    def test_previous_of_first_item_is_none(self):
        first = self.tree.find('html')
        assert first.previous_element == None

    def test_previous_of_root_is_none(self):
        assert self.tree.previous_element == None

    def test_find_all_previous(self):
        self.assert_selects(self.end.find_all_previous('b'), ['Three', 'Two', 'One'])
        self.assert_selects(self.end.find_all_previous(id=1), ['One'])

    def test_find_previous(self):
        assert self.end.find_previous('b')['id'] == '3'
        assert self.end.find_previous(string='One') == 'One'

    def test_find_previous_for_text_element(self):
        text = self.tree.find(string='Three')
        assert text.find_previous('b').string == 'Three'
        self.assert_selects(text.find_all_previous('b'), ['Three', 'Two', 'One'])

    def test_previous_generator(self):
        start = self.tree.find(string='One')
        predecessors = [node for node in start.previous_elements]
        b, body, head, html = predecessors
        assert b['id'] == '1'
        assert body.name == 'body'
        assert head.name == 'head'
        assert html.name == 'html'
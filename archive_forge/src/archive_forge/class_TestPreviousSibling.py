from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestPreviousSibling(SiblingTest):

    def setup_method(self):
        super(TestPreviousSibling, self).setup_method()
        self.end = self.tree.find(id='4')

    def test_previous_sibling_of_root_is_none(self):
        assert self.tree.previous_sibling == None

    def test_previous_sibling(self):
        assert self.end.previous_sibling['id'] == '3'
        assert self.end.previous_sibling.previous_sibling['id'] == '2'
        assert self.end.previous_element['id'] == '3.1'

    def test_previous_sibling_may_not_exist(self):
        assert self.tree.html.previous_sibling == None
        nested_span = self.tree.find(id='1.1')
        assert nested_span.previous_sibling == None
        first_span = self.tree.find(id='1')
        assert first_span.previous_sibling == None

    def test_find_previous_sibling(self):
        assert self.end.find_previous_sibling('span')['id'] == '3'

    def test_previous_siblings(self):
        self.assert_selects_ids(self.end.find_previous_siblings('span'), ['3', '2', '1'])
        self.assert_selects_ids(self.end.find_previous_siblings(id='1'), ['1'])

    def test_previous_sibling_for_text_element(self):
        soup = self.soup('Foo<b>bar</b>baz')
        start = soup.find(string='baz')
        assert start.previous_sibling.name == 'b'
        assert start.previous_sibling.previous_sibling == 'Foo'
        self.assert_selects(start.find_previous_siblings('b'), ['bar'])
        assert start.find_previous_sibling(string='Foo') == 'Foo'
        assert start.find_previous_sibling(string='nonesuch') == None
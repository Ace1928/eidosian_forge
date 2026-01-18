import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
def test_out_of_range_entity(self):
    self.assert_soup('<p>foo&#10000000000000;bar</p>', '<p>foobar</p>')
    self.assert_soup('<p>foo&#x10000000000000;bar</p>', '<p>foobar</p>')
    self.assert_soup('<p>foo&#1000000000;bar</p>', '<p>foobar</p>')
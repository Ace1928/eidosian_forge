import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
def test_pickle_restores_builder(self):
    soup = self.soup('<a>some markup</a>')
    assert isinstance(soup.builder, self.default_builder)
    pickled = pickle.dumps(soup)
    unpickled = pickle.loads(pickled)
    assert 'some markup' == unpickled.a.string
    assert unpickled.builder != soup.builder
    assert isinstance(unpickled.builder, self.default_builder)
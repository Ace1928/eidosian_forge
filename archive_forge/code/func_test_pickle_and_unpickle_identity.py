import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_pickle_and_unpickle_identity(self):
    tree = self.soup('<a><b>foo</a>')
    dumped = pickle.dumps(tree, 2)
    loaded = pickle.loads(dumped)
    assert loaded.__class__ == BeautifulSoup
    assert loaded.decode() == tree.decode()
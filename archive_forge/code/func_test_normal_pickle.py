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
def test_normal_pickle(self):
    soup = self.soup('<a>some markup</a>')
    pickled = pickle.dumps(soup)
    unpickled = pickle.loads(pickled)
    assert 'some markup' == unpickled.a.string
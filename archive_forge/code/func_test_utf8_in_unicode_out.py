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
def test_utf8_in_unicode_out(self):
    soup_from_utf8 = self.soup(self.utf8_data)
    assert soup_from_utf8.decode() == self.unicode_data
    assert soup_from_utf8.foo.string == 'Sacré bleu!'
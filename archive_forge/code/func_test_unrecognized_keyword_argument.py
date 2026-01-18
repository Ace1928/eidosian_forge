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
def test_unrecognized_keyword_argument(self):
    with pytest.raises(TypeError):
        self.soup('<a>', no_such_argument=True)
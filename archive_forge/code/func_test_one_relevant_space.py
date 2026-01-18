import importlib
import codecs
import time
import unicodedata
import pytest
import numpy as np
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern
from . import util
from numpy.f2py import crackfortran
import textwrap
import contextlib
import io
def test_one_relevant_space(self):
    assert markinnerspaces("a 'b c' \\' \\'") == "a 'b@_@c' \\' \\'"
    assert markinnerspaces('a "b c" \\" \\"') == 'a "b@_@c" \\" \\"'
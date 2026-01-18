import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_temppath():
    with temppath() as fpath:
        with open(fpath, 'w'):
            pass
    assert_(not os.path.isfile(fpath))
    raised = False
    try:
        with temppath() as fpath:
            raise ValueError()
    except ValueError:
        raised = True
    assert_(raised)
    assert_(not os.path.isfile(fpath))
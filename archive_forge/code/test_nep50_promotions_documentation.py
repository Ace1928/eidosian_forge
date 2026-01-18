import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM

This file adds basic tests to test the NEP 50 style promotion compatibility
mode.  Most of these test are likely to be simply deleted again once NEP 50
is adopted in the main test suite.  A few may be moved elsewhere.

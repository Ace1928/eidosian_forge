import code
import platform
import pytest
import sys
from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_MUSL
def userinput():
    yield 'np.sqrt(2)'
    raise EOFError
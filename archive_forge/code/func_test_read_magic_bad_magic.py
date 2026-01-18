import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_read_magic_bad_magic():
    for magic in malformed_magic:
        f = BytesIO(magic)
        assert_raises(ValueError, format.read_array, f)
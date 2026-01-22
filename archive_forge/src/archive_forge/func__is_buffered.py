import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def _is_buffered(iterator):
    try:
        iterator.itviews
    except ValueError:
        return True
    return False
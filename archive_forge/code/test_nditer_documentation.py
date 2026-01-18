import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (

    Matches the expected output of a debug print with the actual output.
    Note that the iterator dump should not be considered stable API,
    this test is mainly to ensure the print does not crash.

    Currently uses a subprocess to avoid dealing with the C level `printf`s.
    
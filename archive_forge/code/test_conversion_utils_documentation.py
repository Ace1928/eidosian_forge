import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
Takes valid non-deprecated inputs for converters,
        runs converters on inputs, checks correctness of outputs,
        warnings and errors
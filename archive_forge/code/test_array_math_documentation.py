import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
Tests remainder and mod, which, unlike the C/matlab equivalents, are identical in numpy.
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
The addition method is special for the scaled float, because it
        includes the "cast" between different factors, thus cast-safety
        is influenced by the implementation.
        
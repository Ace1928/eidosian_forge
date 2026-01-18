from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ..affines import (
from ..eulerangles import euler2mat
from ..orientations import aff2axcodes
Check the calculation of inclination of an affine axes.
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
class TestAssignValues_2_UCS4(AssignValues):
    """Check the assignment of valued arrays (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value
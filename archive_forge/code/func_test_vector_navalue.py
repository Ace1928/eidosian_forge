import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
@pytest.mark.parametrize('cls,expected_na', [(robjects.StrVector, ri.NA_Character), (robjects.IntVector, ri.NA_Integer), (robjects.FloatVector, ri.NA_Real), (robjects.BoolVector, ri.NA_Logical), (robjects.ComplexVector, ri.NA_Complex)])
def test_vector_navalue(cls, expected_na):
    assert cls.NAvalue is expected_na
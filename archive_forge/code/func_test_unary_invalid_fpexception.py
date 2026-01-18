import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin_name', ['rint', 'trunc', 'ceil', 'floor'])
def test_unary_invalid_fpexception(self, intrin_name):
    intrin = getattr(self, intrin_name)
    for d in [float('nan'), float('inf'), -float('inf')]:
        v = self.setall(d)
        clear_floatstatus()
        intrin(v)
        assert check_floatstatus(invalid=True) == False
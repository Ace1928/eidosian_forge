import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin', ['any', 'all'])
@pytest.mark.parametrize('data', ([1, 2, 3, 4], [-1, -2, -3, -4], [0, 1, 2, 3, 4], [127, 32767, 2147483647, 9223372036854775807], [0, -1, -2, -3, 4], [0], [1], [-1]))
def test_operators_crosstest(self, intrin, data):
    """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
    data_a = self.load(data * self.nlanes)
    func = eval(intrin)
    intrin = getattr(self, intrin)
    desired = func(data_a)
    simd = intrin(data_a)
    assert not not simd == desired
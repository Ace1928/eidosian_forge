import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
@pytest.mark.parametrize('args', [('Foo', ['a', 'b', 'a'], ['x']), ('Foo', ['a', 'b'], ['b', 'x'])])
def test_repeated_field_names(self, args):
    with pytest.raises(ValueError, match='Duplicate'):
        _make_tuple_bunch(*args)
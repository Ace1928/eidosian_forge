import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def test_rankdata_object_string(self):

    def min_rank(a):
        return [1 + sum((i < j for i in a)) for j in a]

    def max_rank(a):
        return [sum((i <= j for i in a)) for j in a]

    def ordinal_rank(a):
        return min_rank([(x, i) for i, x in enumerate(a)])

    def average_rank(a):
        return [(i + j) / 2.0 for i, j in zip(min_rank(a), max_rank(a))]

    def dense_rank(a):
        b = np.unique(a)
        return [1 + sum((i < j for i in b)) for j in a]
    rankf = dict(min=min_rank, max=max_rank, ordinal=ordinal_rank, average=average_rank, dense=dense_rank)

    def check_ranks(a):
        for method in ('min', 'max', 'dense', 'ordinal', 'average'):
            out = rankdata(a, method=method)
            assert_array_equal(out, rankf[method](a))
    val = ['foo', 'bar', 'qux', 'xyz', 'abc', 'efg', 'ace', 'qwe', 'qaz']
    check_ranks(np.random.choice(val, 200))
    check_ranks(np.random.choice(val, 200).astype('object'))
    val = np.array([0, 1, 2, 2.718, 3, 3.141], dtype='object')
    check_ranks(np.random.choice(val, 200).astype('object'))
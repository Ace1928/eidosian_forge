import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(3, range(5))) == [(0, 1, 2), (3, 4)]
    assert list(partition_all(2, [])) == []

    class NoCompare(object):

        def __eq__(self, other):
            if self.__class__ == other.__class__:
                return True
            raise ValueError()
    obj = NoCompare()
    result = [(obj, obj, obj, obj), (obj, obj, obj)]
    assert list(partition_all(4, [obj] * 7)) == result
    assert list(partition_all(4, iter([obj] * 7))) == result
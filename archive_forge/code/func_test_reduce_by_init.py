import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_reduce_by_init():
    assert reduceby(iseven, add, [1, 2, 3, 4]) == {True: 2 + 4, False: 1 + 3}
    assert reduceby(iseven, add, [1, 2, 3, 4], no_default2) == {True: 2 + 4, False: 1 + 3}
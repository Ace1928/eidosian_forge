import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_sliding_window_of_short_iterator():
    assert list(sliding_window(3, [1, 2])) == []
    assert list(sliding_window(7, [1, 2])) == []
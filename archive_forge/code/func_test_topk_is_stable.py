import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_topk_is_stable():
    assert topk(4, [5, 9, 2, 1, 5, 3], key=lambda x: 1) == (5, 9, 2, 1)
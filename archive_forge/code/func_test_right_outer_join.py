import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_right_outer_join():
    result = set(join(identity, [1, 2], identity, [2, 3], right_default=None))
    expected = {(2, 2), (1, None)}
    assert result == expected
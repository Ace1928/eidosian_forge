import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
class NotIterableEvenWithGetItem:
    __iter__ = None

    def __getitem__(self, item):
        return ['a', 'b', 'c'][item]
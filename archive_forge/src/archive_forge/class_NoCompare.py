import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
class NoCompare(object):

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return True
        raise ValueError()
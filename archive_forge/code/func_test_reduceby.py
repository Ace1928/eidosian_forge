import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_reduceby():
    data = [1, 2, 3, 4, 5]
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, add, data, 0) == {False: 9, True: 6}
    assert reduceby(iseven, mul, data, 1) == {False: 15, True: 8}
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000}, {'name': 'fight crime', 'state': 'IL', 'cost': 100000}, {'name': 'help farmers', 'state': 'IL', 'cost': 2000000}, {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    assert reduceby(lambda x: x['state'], lambda acc, x: acc + x['cost'], projects, 0) == {'CA': 1200000, 'IL': 2100000}
    assert reduceby('state', lambda acc, x: acc + x['cost'], projects, 0) == {'CA': 1200000, 'IL': 2100000}
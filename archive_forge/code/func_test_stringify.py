from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_stringify():
    obj = 'Hello'
    assert stringify(obj) is obj
    obj = b'Hello'
    assert stringify(obj) is obj
    dsk = {'x': 1}
    assert stringify(dsk) == str(dsk)
    assert stringify(dsk, exclusive=()) == dsk
    dsk = {('x', 1): (inc, 1)}
    assert stringify(dsk) == str({('x', 1): (inc, 1)})
    assert stringify(dsk, exclusive=()) == {('x', 1): (inc, 1)}
    dsk = {('x', 1): (inc, 1), ('x', 2): (inc, ('x', 1))}
    assert stringify(dsk, exclusive=dsk) == {('x', 1): (inc, 1), ('x', 2): (inc, str(('x', 1)))}
    dsks = [{'x': 1}, {('x', 1): (inc, 1), ('x', 2): (inc, ('x', 1))}, {('x', 1): (sum, [1, 2, 3]), ('x', 2): (sum, [('x', 1), ('x', 1)])}]
    for dsk in dsks:
        sdsk = {stringify(k): stringify(v, exclusive=dsk) for k, v in dsk.items()}
        keys = list(dsk)
        skeys = [str(k) for k in keys]
        assert all((isinstance(k, str) for k in sdsk))
        assert get(dsk, keys) == get(sdsk, skeys)
    dsk = {('y', 1): (SubgraphCallable({'x': ('y', 1)}, 'x', (('y', 1),)), (('z', 1),))}
    dsk = stringify(dsk, exclusive=set(dsk) | {('z', 1)})
    assert dsk['y', 1][0].dsk['x'] == "('y', 1)"
    assert dsk['y', 1][1][0] == "('z', 1)"
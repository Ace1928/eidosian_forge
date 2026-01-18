import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_returntypes(self):
    qux_fields = {'C': (np.dtype('S5'), 0), 'D': (np.dtype('S5'), 6)}
    a = np.rec.array([('abc ', (1, 1), 1, ('abcde', 'fgehi')), ('abc', (2, 3), 1, ('abcde', 'jklmn'))], dtype=[('foo', 'S4'), ('bar', [('A', int), ('B', int)]), ('baz', int), ('qux', qux_fields)])
    assert_equal(type(a.foo), np.ndarray)
    assert_equal(type(a['foo']), np.ndarray)
    assert_equal(type(a.bar), np.recarray)
    assert_equal(type(a['bar']), np.recarray)
    assert_equal(a.bar.dtype.type, np.record)
    assert_equal(type(a['qux']), np.recarray)
    assert_equal(a.qux.dtype.type, np.record)
    assert_equal(dict(a.qux.dtype.fields), qux_fields)
    assert_equal(type(a.baz), np.ndarray)
    assert_equal(type(a['baz']), np.ndarray)
    assert_equal(type(a[0].bar), np.record)
    assert_equal(type(a[0]['bar']), np.record)
    assert_equal(a[0].bar.A, 1)
    assert_equal(a[0].bar['A'], 1)
    assert_equal(a[0]['bar'].A, 1)
    assert_equal(a[0]['bar']['A'], 1)
    assert_equal(a[0].qux.D, b'fgehi')
    assert_equal(a[0].qux['D'], b'fgehi')
    assert_equal(a[0]['qux'].D, b'fgehi')
    assert_equal(a[0]['qux']['D'], b'fgehi')
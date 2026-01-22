from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
class TestDictionaries:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = {'a': 'A', 'b': 'B'}
        self.y = {'c': 'C', 'b': 'B'}
        self.z = {'a': 'Z'}

    def test_equivalent(self):
        assert utils.equivalent(0, 0)
        assert utils.equivalent(np.nan, np.nan)
        assert utils.equivalent(0, np.array(0.0))
        assert utils.equivalent([0], np.array([0]))
        assert utils.equivalent(np.array([0]), [0])
        assert utils.equivalent(np.arange(3), 1.0 * np.arange(3))
        assert not utils.equivalent(0, np.zeros(3))

    def test_safe(self):
        utils.update_safety_check(self.x, self.y)

    def test_unsafe(self):
        with pytest.raises(ValueError):
            utils.update_safety_check(self.x, self.z)

    def test_compat_dict_intersection(self):
        assert {'b': 'B'} == utils.compat_dict_intersection(self.x, self.y)
        assert {} == utils.compat_dict_intersection(self.x, self.z)

    def test_compat_dict_union(self):
        assert {'a': 'A', 'b': 'B', 'c': 'C'} == utils.compat_dict_union(self.x, self.y)
        with pytest.raises(ValueError, match='unsafe to merge dictionaries without overriding values; conflicting key'):
            utils.compat_dict_union(self.x, self.z)

    def test_dict_equiv(self):
        x = {}
        x['a'] = 3
        x['b'] = np.array([1, 2, 3])
        y = {}
        y['b'] = np.array([1.0, 2.0, 3.0])
        y['a'] = 3
        assert utils.dict_equiv(x, y)
        y['b'] = [1, 2, 3]
        assert utils.dict_equiv(x, y)
        x['b'] = [1.0, 2.0, 3.0]
        assert utils.dict_equiv(x, y)
        x['c'] = None
        assert not utils.dict_equiv(x, y)
        x['c'] = np.nan
        y['c'] = np.nan
        assert utils.dict_equiv(x, y)
        x['c'] = np.inf
        y['c'] = np.inf
        assert utils.dict_equiv(x, y)
        y = dict(y)
        assert utils.dict_equiv(x, y)
        y['b'] = 3 * np.arange(3)
        assert not utils.dict_equiv(x, y)

    def test_frozen(self):
        x = utils.Frozen(self.x)
        with pytest.raises(TypeError):
            x['foo'] = 'bar'
        with pytest.raises(TypeError):
            del x['a']
        with pytest.raises(AttributeError):
            x.update(self.y)
        assert x.mapping == self.x
        assert repr(x) in ("Frozen({'a': 'A', 'b': 'B'})", "Frozen({'b': 'B', 'a': 'A'})")
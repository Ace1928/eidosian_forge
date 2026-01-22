from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
class TestRaggedInterface(eb.BaseInterfaceTests):

    def test_array_interface(self, data):
        result = np.array(data, dtype=object)
        np.testing.assert_array_equal(result[0], data[0])
        result = np.array(data, dtype=object)
        expected = np.array(list(data), dtype=object)
        for a1, a2 in zip(result, expected):
            if np.isscalar(a1):
                assert np.isnan(a1) and np.isnan(a2)
            else:
                np.testing.assert_array_equal(a1, a2)

    @pytest.mark.skip(reason='__setitem__ not supported')
    def test_copy(self):
        pass

    @pytest.mark.skip(reason='__setitem__ not supported')
    def test_view(self):
        pass

    @pytest.mark.skipif(Version(pd.__version__) < Version('1.4'), reason='Added in pandas 1.4')
    def test_tolist(self, data):
        result = data.tolist()
        expected = list(data)
        assert isinstance(result, list)
        for r, e in zip(result, expected):
            assert np.array_equal(r, e, equal_nan=True)
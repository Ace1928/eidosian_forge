from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
class TestRaggedReshaping(eb.BaseReshapingTests):

    @pytest.mark.skip(reason='__setitem__ not supported')
    def test_ravel(self):
        pass

    @pytest.mark.skip(reason='transpose with numpy array elements seems not supported')
    def test_transpose(self):
        pass

    @pytest.mark.skip(reason='transpose with numpy array elements seems not supported')
    def test_transpose_frame(self):
        pass

    @pytest.mark.skipif(Version(pd.__version__) == Version('2.2.0'), reason='Regression in Pandas 2.2')
    def test_merge_on_extension_array(self, data):
        super().test_merge_on_extension_array(data)

    @pytest.mark.skipif(Version(pd.__version__) == Version('2.2.0'), reason='Regression in Pandas 2.2')
    def test_merge_on_extension_array_duplicates(self, data):
        super().test_merge_on_extension_array_duplicates(data)
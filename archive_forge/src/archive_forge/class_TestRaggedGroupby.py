from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
class TestRaggedGroupby(eb.BaseGroupbyTests):

    @pytest.mark.skip(reason='agg not supported')
    def test_groupby_agg_extension(self):
        pass

    @pytest.mark.skip(reason='numpy.ndarray unhashable')
    def test_groupby_extension_transform(self):
        pass

    @pytest.mark.skip(reason='agg not supported')
    def test_groupby_extension_agg(self):
        pass

    @pytest.mark.skip(reason='numpy.ndarray unhashable and buffer wrong number of dims')
    def test_groupby_extension_apply(self):
        pass
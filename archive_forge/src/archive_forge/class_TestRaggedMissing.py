from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
class TestRaggedMissing(eb.BaseMissingTests):

    @pytest.mark.skip(reason="Can't fill with ndarray")
    def test_fillna_series(self):
        pass

    @pytest.mark.skip(reason="Can't fill with ndarray")
    def test_fillna_frame(self):
        pass

    @pytest.mark.skip(reason="Can't fill with nested sequences")
    def test_fillna_limit_pad(self):
        pass

    @pytest.mark.skip(reason="Can't fill with nested sequences")
    def test_fillna_limit_backfill(self):
        pass

    @pytest.mark.skip(reason="Can't fill with nested sequences")
    def test_fillna_no_op_returns_copy(self):
        pass

    @pytest.mark.skip(reason="Can't set array element with a sequence")
    def test_fillna_series_method(self):
        pass

    @pytest.mark.skip(reason="Can't fill with nested sequences")
    def test_ffill_limit_area(self):
        pass
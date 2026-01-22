from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
class TestRaggedPrinting(eb.BasePrintingTests):

    @pytest.mark.skip(reason="Can't autoconvert ragged array to numpy array")
    def test_dataframe_repr(self):
        pass

    @pytest.mark.skip(reason="Can't autoconvert ragged array to numpy array")
    def test_series_repr(self):
        pass
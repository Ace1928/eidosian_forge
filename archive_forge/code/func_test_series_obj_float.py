import math
from collections import OrderedDict
from datetime import datetime
import pytest
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects import conversion
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
@pytest.mark.parametrize('data', ([1.1, 2.2, 3.3], [1.1, 2.2, None]))
@pytest.mark.parametrize('dtype', [float, pandas.Float64Dtype() if has_pandas else None])
def test_series_obj_float(self, data, dtype):
    Series = pandas.core.series.Series
    s = Series(data, index=['a', 'b', 'c'], dtype=dtype)
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_s = robjects.conversion.converter_ctx.get().py2rpy(s)
    assert isinstance(rp_s, rinterface.FloatSexpVector)
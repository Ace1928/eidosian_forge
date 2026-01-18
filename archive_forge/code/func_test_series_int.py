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
@pytest.mark.parametrize('dtype', ('i', numpy.int32 if has_pandas else None, numpy.int8 if has_pandas else None, numpy.int16 if has_pandas else None, numpy.int32 if has_pandas else None, numpy.int64 if has_pandas else None, numpy.uint8 if has_pandas else None, numpy.uint16 if has_pandas else None, pandas.Int32Dtype if has_pandas else None, pandas.Int64Dtype if has_pandas else None))
def test_series_int(self, dtype):
    Series = pandas.core.series.Series
    s = Series(range(5), index=['a', 'b', 'c', 'd', 'e'], dtype=dtype)
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_s = robjects.conversion.get_conversion().py2rpy(s)
    assert isinstance(rp_s, rinterface.IntSexpVector)
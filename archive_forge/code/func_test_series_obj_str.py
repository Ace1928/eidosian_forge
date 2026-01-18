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
@pytest.mark.skipif(not (has_numpy and has_pandas), reason='Packages numpy and pandas must be installed.')
@pytest.mark.parametrize('data', (['x', 'y', 'z'], ['x', 'y', None], ['x', 'y', numpy.nan], ['x', 'y', pandas.NA]))
@pytest.mark.parametrize('dtype', ['O', pandas.StringDtype() if has_pandas else None])
def test_series_obj_str(self, data, dtype):
    Series = pandas.core.series.Series
    s = Series(data, index=['a', 'b', 'c'], dtype=dtype)
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_s = robjects.conversion.converter_ctx.get().py2rpy(s)
    assert isinstance(rp_s, rinterface.StrSexpVector)
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
@pytest.mark.skipif(not has_pandas, reason='pandas must be installed.')
@pytest.mark.parametrize('data', ([True, False, True], [True, False, None]))
@pytest.mark.parametrize('dtype', [bool, pandas.BooleanDtype() if has_pandas else None])
@pytest.mark.parametrize('constructor,wrapcheck', [(pandas.Series, lambda x: x), (pandas.DataFrame, lambda x: x[0])])
def test_series_obj_bool(self, data, dtype, constructor, wrapcheck):
    s = constructor(data, index=['a', 'b', 'c'], dtype=dtype)
    with localconverter(default_converter + rpyp.converter) as cv:
        rp_s = robjects.conversion.converter_ctx.get().py2rpy(s)
    assert isinstance(wrapcheck(rp_s), rinterface.BoolSexpVector)
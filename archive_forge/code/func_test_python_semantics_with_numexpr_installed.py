import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.mark.parametrize('op', ['__mod__', '__rmod__', '__floordiv__', '__rfloordiv__'])
@pytest.mark.parametrize('box', [DataFrame, Series, Index])
@pytest.mark.parametrize('scalar', [-5, 5])
def test_python_semantics_with_numexpr_installed(self, op, box, scalar, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', 0)
        data = np.arange(-50, 50)
        obj = box(data)
        method = getattr(obj, op)
        result = method(scalar)
        with option_context('compute.use_numexpr', False):
            expected = method(scalar)
        tm.assert_equal(result, expected)
        for i, elem in enumerate(data):
            if box == DataFrame:
                scalar_result = result.iloc[i, 0]
            else:
                scalar_result = result[i]
            try:
                expected = getattr(int(elem), op)(scalar)
            except ZeroDivisionError:
                pass
            else:
                assert scalar_result == expected
import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
@pytest.mark.parametrize('flex', [True, False])
def test_run_binary(self, request, fixture, flex, comparison_op, monkeypatch):
    """
        tests solely that the result is the same whether or not numexpr is
        enabled.  Need to test whether the function does the correct thing
        elsewhere.
        """
    df = request.getfixturevalue(fixture)
    arith = comparison_op.__name__
    with option_context('compute.use_numexpr', False):
        other = df.copy() + 1
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', 0)
        expr.set_test_mode(True)
        result, expected = self.call_op(df, other, flex, arith)
        used_numexpr = expr.get_test_result()
        assert used_numexpr, 'Did not use numexpr as expected.'
        tm.assert_equal(expected, result)
        for i in range(len(df.columns)):
            binary_comp = other.iloc[:, i] + 1
            self.call_op(df.iloc[:, i], binary_comp, flex, 'add')
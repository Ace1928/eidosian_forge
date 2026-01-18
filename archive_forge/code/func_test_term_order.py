import sys
import __future__
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc, Term, INTERCEPT
from patsy.categorical import C
from patsy.contrasts import Helmert
from patsy.user_util import balanced, LookupFactor
from patsy.build import (design_matrix_builders,
from patsy.highlevel import *
from patsy.util import (have_pandas,
from patsy.origin import Origin
def test_term_order():
    data = balanced(a=2, b=2)
    data['x1'] = np.linspace(0, 1, 4)
    data['x2'] = data['x1'] ** 2

    def t_terms(formula, order):
        m = dmatrix(formula, data)
        assert m.design_info.term_names == order
    t_terms('a + b + x1 + x2', ['Intercept', 'a', 'b', 'x1', 'x2'])
    t_terms('b + a + x2 + x1', ['Intercept', 'b', 'a', 'x2', 'x1'])
    t_terms('0 + x1 + a + x2 + b + 1', ['Intercept', 'a', 'b', 'x1', 'x2'])
    t_terms('0 + a:b + a + b + 1', ['Intercept', 'a', 'b', 'a:b'])
    t_terms('a + a:x1 + x2 + x1 + b', ['Intercept', 'a', 'b', 'x1', 'a:x1', 'x2'])
    t_terms('0 + a:x1:x2 + a + x2:x1:b + x2 + x1 + a:x1 + x1:x2 + x1:a:x2:a:b', ['a', 'x1:x2', 'a:x1:x2', 'x2:x1:b', 'x1:a:x2:b', 'x2', 'x1', 'a:x1'])
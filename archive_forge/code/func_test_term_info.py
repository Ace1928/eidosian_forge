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
def test_term_info():
    data = balanced(a=2, b=2)
    rhs = dmatrix('a:b', data)
    assert rhs.design_info.column_names == ['Intercept', 'b[T.b2]', 'a[T.a2]:b[b1]', 'a[T.a2]:b[b2]']
    assert rhs.design_info.term_names == ['Intercept', 'a:b']
    assert len(rhs.design_info.terms) == 2
    assert rhs.design_info.terms[0] == INTERCEPT
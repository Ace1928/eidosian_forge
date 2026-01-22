import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def _build_design_matrix(design_info, factor_info_to_values, dtype):
    factor_to_values = {}
    need_reshape = False
    num_rows = None
    for factor_info, value in six.iteritems(factor_info_to_values):
        if design_info.factor_infos.get(factor_info.factor) is not factor_info:
            continue
        factor_to_values[factor_info.factor] = value
        if num_rows is not None:
            assert num_rows == value.shape[0]
        else:
            num_rows = value.shape[0]
    if num_rows is None:
        num_rows = 1
        need_reshape = True
    shape = (num_rows, len(design_info.column_names))
    m = DesignMatrix(np.empty(shape, dtype=dtype), design_info)
    start_column = 0
    for term, subterms in six.iteritems(design_info.term_codings):
        for subterm in subterms:
            end_column = start_column + subterm.num_columns
            m_slice = m[:, start_column:end_column]
            _build_subterm(subterm, design_info.factor_infos, factor_to_values, m_slice)
            start_column = end_column
    assert start_column == m.shape[1]
    return (need_reshape, m)
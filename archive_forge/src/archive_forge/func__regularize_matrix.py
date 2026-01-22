import six
import numpy as np
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc
from patsy.build import (design_matrix_builders,
from patsy.util import (have_pandas, asarray_or_pandas,
def _regularize_matrix(m, default_column_prefix):
    di = DesignInfo.from_array(m, default_column_prefix)
    if have_pandas and isinstance(m, (pandas.Series, pandas.DataFrame)):
        orig_index = m.index
    else:
        orig_index = None
    if return_type == 'dataframe':
        m = atleast_2d_column_default(m, preserve_pandas=True)
        m = pandas.DataFrame(m)
        m.columns = di.column_names
        m.design_info = di
        return (m, orig_index)
    else:
        return (DesignMatrix(m, di), orig_index)
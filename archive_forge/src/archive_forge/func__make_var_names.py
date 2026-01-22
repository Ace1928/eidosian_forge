from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _make_var_names(exog):
    if hasattr(exog, 'name'):
        var_names = [exog.name]
    elif hasattr(exog, 'columns'):
        var_names = exog.columns
    else:
        raise ValueError('exog is not a Series or DataFrame or is unnamed.')
    try:
        var_names = ' '.join(var_names)
    except TypeError:
        from statsmodels.base.data import _make_exog_names
        if exog.ndim == 1:
            var_names = 'x1'
        else:
            var_names = ' '.join(_make_exog_names(exog))
    return var_names
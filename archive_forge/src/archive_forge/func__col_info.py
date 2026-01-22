from statsmodels.compat.pandas import FUTURE_STACK
from statsmodels.compat.python import lzip
import datetime
from functools import reduce
import re
import textwrap
import numpy as np
import pandas as pd
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt
def _col_info(result, info_dict=None):
    """Stack model info in a column
    """
    if info_dict is None:
        info_dict = {}
    out = []
    index = []
    for i in info_dict:
        if isinstance(info_dict[i], dict):
            continue
        try:
            out.append(info_dict[i](result))
        except AttributeError:
            out.append('')
        index.append(i)
    out = pd.DataFrame({str(result.model.endog_names): out}, index=index)
    return out
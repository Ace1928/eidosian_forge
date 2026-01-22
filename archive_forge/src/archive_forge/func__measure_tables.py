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
def _measure_tables(tables, settings):
    """Compare width of ascii tables in a list and calculate padding values.
    We add space to each col_sep to get us as close as possible to the
    width of the largest table. Then, we add a few spaces to the first
    column to pad the rest.
    """
    simple_tables = _simple_tables(tables, settings)
    tab = [x.as_text() for x in simple_tables]
    length = [len(x.splitlines()[0]) for x in tab]
    len_max = max(length)
    pad_sep = []
    pad_index = []
    for i in range(len(tab)):
        nsep = max(tables[i].shape[1] - 1, 1)
        pad = int((len_max - length[i]) / nsep)
        pad_sep.append(pad)
        len_new = length[i] + nsep * pad
        pad_index.append(len_max - len_new)
    return (pad_sep, pad_index, max(length))
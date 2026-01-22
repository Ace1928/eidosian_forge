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
def add_title(self, title=None, results=None):
    """Insert a title on top of the summary table. If a string is provided
        in the title argument, that string is printed. If no title string is
        provided but a results instance is provided, statsmodels attempts
        to construct a useful title automatically.
        """
    if isinstance(title, str):
        self.title = title
    elif results is not None:
        model = results.model.__class__.__name__
        if model in _model_types:
            model = _model_types[model]
        self.title = 'Results: ' + model
    else:
        self.title = ''
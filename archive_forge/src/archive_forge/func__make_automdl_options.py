from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _make_automdl_options(maxorder, maxdiff, diff):
    options = '\n'
    options += f'maxorder = ({maxorder[0]} {maxorder[1]})\n'
    if maxdiff is not None:
        options += f'maxdiff = ({maxdiff[0]} {maxdiff[1]})\n'
    else:
        options += f'diff = ({diff[0]} {diff[1]})\n'
    return options
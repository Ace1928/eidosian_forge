from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _check_x12(x12path=None):
    x12path = _find_x12(x12path)
    if not x12path:
        raise X13NotFoundError('x12a and x13as not found on path. Give the path, put them on PATH, or set the X12PATH or X13PATH environmental variable.')
    return x12path
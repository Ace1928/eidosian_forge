from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _find_x12(x12path=None, prefer_x13=True):
    """
    If x12path is not given, then either x13as[.exe] or x12a[.exe] must
    be found on the PATH. Otherwise, the environmental variable X12PATH or
    X13PATH must be defined. If prefer_x13 is True, only X13PATH is searched
    for. If it is false, only X12PATH is searched for.
    """
    global _binary_names
    if x12path is not None and x12path.endswith(_binary_names):
        if not os.path.isdir(x12path):
            x12path = os.path.dirname(x12path)
    if not prefer_x13:
        _binary_names = _binary_names[::-1]
        if x12path is None:
            x12path = os.getenv('X12PATH', '')
        if not x12path:
            x12path = os.getenv('X13PATH', '')
    elif x12path is None:
        x12path = os.getenv('X13PATH', '')
        if not x12path:
            x12path = os.getenv('X12PATH', '')
    for binary in _binary_names:
        x12 = os.path.join(x12path, binary)
        try:
            subprocess.check_call(x12, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return x12
        except OSError:
            pass
    else:
        return False
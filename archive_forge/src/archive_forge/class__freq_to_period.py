from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
class _freq_to_period:

    def __getitem__(self, key):
        if key.startswith('M'):
            return 12
        elif key.startswith('Q'):
            return 4
        elif key.startswith('W'):
            return 52
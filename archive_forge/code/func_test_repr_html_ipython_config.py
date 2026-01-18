from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_html_ipython_config(self, ip):
    code = textwrap.dedent('        from pandas import DataFrame\n        df = DataFrame({"A": [1, 2]})\n        df._repr_html_()\n\n        cfg = get_ipython().config\n        cfg[\'IPKernelApp\'][\'parent_appname\']\n        df._repr_html_()\n        ')
    result = ip.run_cell(code, silent=True)
    assert not result.error_in_exec
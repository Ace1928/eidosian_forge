import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.single_cpu
def test_oo_optimizable():
    subprocess.check_call([sys.executable, '-OO', '-c', 'import pandas'])
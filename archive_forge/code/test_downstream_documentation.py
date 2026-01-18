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

    Test some basic methods of the dataframe consortium standard.

    Full testing is done at https://github.com/data-apis/dataframe-api-compat,
    this is just to check that the entry point works as expected.
    
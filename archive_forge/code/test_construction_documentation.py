import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
Fixture returning parametrized IntegerArray from given sequence.

    Used to test dtype conversions.
    
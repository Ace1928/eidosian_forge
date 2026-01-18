import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm

Tests for values coercion in setitem-like operations on DataFrame.

For the most part, these should be multi-column DataFrames, otherwise
we would share the tests with Series.

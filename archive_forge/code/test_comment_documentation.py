from io import StringIO
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm

Tests that comments are properly handled during parsing
for all of the parsers defined in parsers.py

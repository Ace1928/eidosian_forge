from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm

Tests the usecols functionality during parsing
for all of the parsers defined in parsers.py

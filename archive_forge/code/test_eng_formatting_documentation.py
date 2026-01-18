import numpy as np
import pytest
from pandas import (
from pandas.io.formats.format import EngFormatter

        Parameters:
        -----------
        formatter: EngFormatter under test
        in_out: list of tuples. Each tuple = (number, expected_formatting)

        It is tested if 'formatter(number) == expected_formatting'.
        *number* should be >= 0 because formatter(-number) == fmt is also
        tested. *fmt* is derived from *expected_formatting*
        
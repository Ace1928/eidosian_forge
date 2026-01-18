from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

        Test the formatting of EngFormatter for various values of the 'places'
        argument, in several cases:

        0. without a unit symbol but with a (default) space separator;
        1. with both a unit symbol and a (default) space separator;
        2. with both a unit symbol and some non default separators;
        3. without a unit symbol but with some non default separators.

        Note that cases 2. and 3. are looped over several separator strings.
        
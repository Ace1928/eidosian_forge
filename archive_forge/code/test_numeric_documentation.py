import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
A property-based test using Hypothesis.

        This aims for maximum generality: it could in principle generate *any*
        valid inputs to np.clip, and in practice generates much more varied
        inputs than human testers come up with.

        Because many of the inputs have tricky dependencies - compatible dtypes
        and mutually-broadcastable shapes - we use `st.data()` strategy draw
        values *inside* the test function, from strategies we construct based
        on previous values.  An alternative would be to define a custom strategy
        with `@st.composite`, but until we have duplicated code inline is fine.

        That accounts for most of the function; the actual test is just three
        lines to calculate and compare actual vs expected results!
        
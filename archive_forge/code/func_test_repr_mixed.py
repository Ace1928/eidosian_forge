from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_mixed(self, float_string_frame):
    buf = StringIO()
    repr(float_string_frame)
    float_string_frame.info(verbose=False, buf=buf)
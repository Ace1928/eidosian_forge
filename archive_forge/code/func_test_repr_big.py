from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.slow
def test_repr_big(self):
    biggie = DataFrame(np.zeros((200, 4)), columns=range(4), index=range(200))
    repr(biggie)
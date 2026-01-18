from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
def test_over_specified():
    data = 'A   B     C            D            E\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n201160    364.136849   183.628767   11806.2\n201161    413.836124   184.375703   11916.8\n201162    502.953953   173.237159   12468.3\n'
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    with pytest.raises(ValueError, match='must specify only one of'):
        read_fwf(StringIO(data), colspecs=colspecs, widths=[6, 10, 10, 7])
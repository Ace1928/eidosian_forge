from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('unit', ['100px', '10%', '5em', 150])
def test_to_html_with_col_space_units(unit):
    df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    result = df.to_html(col_space=unit)
    result = result.split('tbody')[0]
    hdrs = [x for x in result.split('\n') if re.search('<th[>\\s]', x)]
    if isinstance(unit, int):
        unit = str(unit) + 'px'
    for h in hdrs:
        expected = f'<th style="min-width: {unit};">'
        assert expected in h
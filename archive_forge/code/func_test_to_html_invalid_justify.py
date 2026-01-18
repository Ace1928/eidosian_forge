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
@pytest.mark.parametrize('justify', ['super-right', 'small-left', 'noinherit', 'tiny', 'pandas'])
def test_to_html_invalid_justify(justify):
    df = DataFrame()
    msg = 'Invalid value for justify parameter'
    with pytest.raises(ValueError, match=msg):
        df.to_html(justify=justify)
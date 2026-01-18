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
@pytest.mark.parametrize('classes', ['sortable draggable', ['sortable', 'draggable']])
def test_to_html_with_classes(classes, datapath):
    df = DataFrame()
    expected = expected_html(datapath, 'with_classes')
    result = df.to_html(classes=classes)
    assert result == expected
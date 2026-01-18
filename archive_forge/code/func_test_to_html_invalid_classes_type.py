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
@pytest.mark.parametrize('classes', [True, 0])
def test_to_html_invalid_classes_type(classes):
    df = DataFrame()
    msg = 'classes must be a string, list, or tuple'
    with pytest.raises(TypeError, match=msg):
        df.to_html(classes=classes)
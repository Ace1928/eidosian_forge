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
def test_to_html_round_column_headers():
    df = DataFrame([1], columns=[0.55555])
    with option_context('display.precision', 3):
        html = df.to_html(notebook=False)
        notebook = df.to_html(notebook=True)
    assert '0.55555' in html
    assert '0.556' in notebook
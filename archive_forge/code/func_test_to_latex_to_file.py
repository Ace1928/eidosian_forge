import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_to_file(self, float_frame):
    with tm.ensure_clean('test.tex') as path:
        float_frame.to_latex(path)
        with open(path, encoding='utf-8') as f:
            assert float_frame.to_latex() == f.read()
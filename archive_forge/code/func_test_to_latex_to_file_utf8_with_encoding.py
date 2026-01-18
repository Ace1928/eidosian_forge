import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_to_file_utf8_with_encoding(self):
    df = DataFrame([['außgangen']])
    with tm.ensure_clean('test.tex') as path:
        df.to_latex(path, encoding='utf-8')
        with codecs.open(path, 'r', encoding='utf-8') as f:
            assert df.to_latex() == f.read()
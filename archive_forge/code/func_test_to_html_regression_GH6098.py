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
def test_to_html_regression_GH6098():
    df = DataFrame({'clé1': ['a', 'a', 'b', 'b', 'a'], 'clé2': ['1er', '2ème', '1er', '2ème', '1er'], 'données1': np.random.default_rng(2).standard_normal(5), 'données2': np.random.default_rng(2).standard_normal(5)})
    df.pivot_table(index=['clé1'], columns=['clé2'])._repr_html_()
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
@pytest.mark.parametrize('render_links,expected', [(True, 'render_links_true'), (False, 'render_links_false')])
def test_to_html_render_links(render_links, expected, datapath):
    data = [[0, 'https://pandas.pydata.org/?q1=a&q2=b', 'pydata.org'], [0, 'www.pydata.org', 'pydata.org']]
    df = DataFrame(data, columns=Index(['foo', 'bar', None], dtype=object))
    result = df.to_html(render_links=render_links)
    expected = expected_html(datapath, expected)
    assert result == expected
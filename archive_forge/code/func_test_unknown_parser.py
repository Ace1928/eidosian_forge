from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_unknown_parser(geom_df):
    with pytest.raises(ValueError, match='Values for parser can only be lxml or etree.'):
        geom_df.to_xml(parser='bs4')
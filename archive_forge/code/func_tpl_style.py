from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.fixture
def tpl_style(env):
    return env.get_template('html_style.tpl')
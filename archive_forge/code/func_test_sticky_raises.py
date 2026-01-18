from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_sticky_raises(styler):
    with pytest.raises(ValueError, match='No axis named bad for object type DataFrame'):
        styler.set_sticky(axis='bad')
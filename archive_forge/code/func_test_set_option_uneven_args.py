import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_set_option_uneven_args(self):
    msg = 'Must provide an even number of non-keyword arguments'
    with pytest.raises(ValueError, match=msg):
        cf.set_option('a.b', 2, 'b.c')
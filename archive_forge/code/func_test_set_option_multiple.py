import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_set_option_multiple(self):
    cf.register_option('a', 1, 'doc')
    cf.register_option('b.c', 'hullo', 'doc2')
    cf.register_option('b.b', None, 'doc2')
    assert cf.get_option('a') == 1
    assert cf.get_option('b.c') == 'hullo'
    assert cf.get_option('b.b') is None
    cf.set_option('a', '2', 'b.c', None, 'b.b', 10.0)
    assert cf.get_option('a') == '2'
    assert cf.get_option('b.c') is None
    assert cf.get_option('b.b') == 10.0
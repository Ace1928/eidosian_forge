import string
import pandas._config.config as cf
from pandas.io.formats import printing
def test_repr_set(self):
    assert printing.pprint_thing({1}) == '{1}'
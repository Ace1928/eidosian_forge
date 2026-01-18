import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclass_attr_err_propagation(self):

    class A(DataFrame):

        @property
        def nonexistence(self):
            return self.i_dont_exist
    with pytest.raises(AttributeError, match='.*i_dont_exist.*'):
        A().nonexistence
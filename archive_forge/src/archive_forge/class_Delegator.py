from datetime import datetime
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
class Delegator:
    _properties = ['prop']
    _methods = ['test_method']

    def _set_prop(self, value):
        self.prop = value

    def _get_prop(self):
        return self.prop
    prop = property(_get_prop, _set_prop, doc='foo property')

    def test_method(self, *args, **kwargs):
        """a test method"""
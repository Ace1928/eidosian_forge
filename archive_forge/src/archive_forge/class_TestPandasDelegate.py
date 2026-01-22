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
class TestPandasDelegate:

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

    class Delegate(PandasDelegate, PandasObject):

        def __init__(self, obj) -> None:
            self.obj = obj

    def test_invalid_delegation(self):
        self.Delegate._add_delegate_accessors(delegate=self.Delegator, accessors=self.Delegator._properties, typ='property')
        self.Delegate._add_delegate_accessors(delegate=self.Delegator, accessors=self.Delegator._methods, typ='method')
        delegate = self.Delegate(self.Delegator())
        msg = 'You cannot access the property prop'
        with pytest.raises(TypeError, match=msg):
            delegate.prop
        msg = 'The property prop cannot be set'
        with pytest.raises(TypeError, match=msg):
            delegate.prop = 5
        msg = 'You cannot access the property prop'
        with pytest.raises(TypeError, match=msg):
            delegate.prop

    @pytest.mark.skipif(PYPY, reason='not relevant for PyPy')
    def test_memory_usage(self):
        delegate = self.Delegate(self.Delegator())
        sys.getsizeof(delegate)
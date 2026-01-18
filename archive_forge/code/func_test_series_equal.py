from unittest import TestCase
from traitlets import HasTraits, TraitError, observe, Undefined
from traitlets.tests.test_traitlets import TraitTestBase
from traittypes import Array, DataFrame, Series, Dataset, DataArray
import numpy as np
import pandas as pd
import xarray as xr
def test_series_equal(self):
    notifications = []

    class Foo(HasTraits):
        bar = Series([1, 2])

        @observe('bar')
        def _(self, change):
            notifications.append(change)
    foo = Foo()
    foo.bar = [1, 2]
    self.assertEqual(notifications, [])
    foo.bar = [1, 1]
    self.assertEqual(len(notifications), 1)
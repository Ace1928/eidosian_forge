from unittest import TestCase
from traitlets import HasTraits, TraitError, observe, Undefined
from traitlets.tests.test_traitlets import TraitTestBase
from traittypes import Array, DataFrame, Series, Dataset, DataArray
import numpy as np
import pandas as pd
import xarray as xr
class TestDataFrame(TestCase):

    def test_df_equal(self):
        notifications = []

        class Foo(HasTraits):
            bar = DataFrame([1, 2])

            @observe('bar')
            def _(self, change):
                notifications.append(change)
        foo = Foo()
        foo.bar = [1, 2]
        self.assertEqual(notifications, [])
        foo.bar = [1, 1]
        self.assertEqual(len(notifications), 1)

    def test_initial_values(self):

        class Foo(HasTraits):
            a = DataFrame()
            b = DataFrame(None, allow_none=True)
            c = DataFrame([])
            d = DataFrame(Undefined)
        foo = Foo()
        self.assertTrue(foo.a.equals(pd.DataFrame()))
        self.assertTrue(foo.b is None)
        self.assertTrue(foo.c.equals(pd.DataFrame([])))
        self.assertTrue(foo.d is Undefined)

    def test_allow_none(self):

        class Foo(HasTraits):
            bar = DataFrame()
            baz = DataFrame(allow_none=True)
        foo = Foo()
        with self.assertRaises(TraitError):
            foo.bar = None
        foo.baz = None
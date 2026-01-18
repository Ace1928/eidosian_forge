import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_sample_ycoord(self):
    xs = np.linspace(-9, 9, 10)
    data = (xs,) + tuple((self.rgb_array[4, :, i] for i in range(3)))
    with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.rgb):
        self.assertEqual(self.rgb.sample(y=5), self.rgb.clone(data, kdims=['x'], new_type=Curve))
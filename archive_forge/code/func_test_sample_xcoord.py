import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_sample_xcoord(self):
    ys = np.linspace(0.5, 9.5, 10)
    data = (ys,) + tuple((self.rgb_array[:, 7, i] for i in range(3)))
    with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.rgb):
        self.assertEqual(self.rgb.sample(x=5), self.rgb.clone(data, kdims=['y'], new_type=Curve))
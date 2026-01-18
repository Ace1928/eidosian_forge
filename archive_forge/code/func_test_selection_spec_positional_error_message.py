import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_selection_spec_positional_error_message(self):
    s, e = ('1999-12-31', '2000-1-2')
    curve = self.datetime_fn()
    with self.assertRaisesRegex(ValueError, 'Use the selection_specs keyword'):
        curve.select((Curve,), time=(s, e))
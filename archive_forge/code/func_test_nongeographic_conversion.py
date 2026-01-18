import unittest
from holoviews.core import HoloMap
from holoviews.element import Curve
from geoviews.element import is_geographic, Image, Dataset
from geoviews.element.comparison import ComparisonTestCase
def test_nongeographic_conversion(self):
    converted = Dataset(self.cube, kdims=['longitude', 'latitude']).to.curve(['longitude'])
    self.assertTrue(isinstance(converted, HoloMap))
    self.assertEqual(converted.kdims, ['latitude'])
    self.assertTrue(isinstance(converted.last, Curve))
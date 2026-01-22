import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
class RasterMapComparisonTest(RasterMapTestCase):

    def test_dimension_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map1_2D)
            raise AssertionError('Mismatch in dimension number not raised.')
        except AssertionError as e:
            self.assertEqual(str(e), 'Key dimension list mismatched')

    def test_dimension_label_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map6_1D)
            raise AssertionError('Mismatch in dimension labels not raised.')
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension names mismatched: int != int_v2')

    def test_key_len_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map3_1D)
            raise AssertionError('Mismatch in map key number not raised.')
        except AssertionError as e:
            self.assertEqual(str(e), 'HoloMaps have different numbers of keys.')

    def test_key_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map2_1D)
            raise AssertionError('Mismatch in map keys not raised.')
        except AssertionError as e:
            self.assertEqual(str(e), 'HoloMaps have different sets of keys. In first, not second [0]. In second, not first: [2].')

    def test_element_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map4_1D)
            raise AssertionError('Pane mismatch in array data not raised.')
        except AssertionError as e:
            if not str(e).startswith('Image not almost equal to 6 decimals\n'):
                raise self.failureException('Image mismatch error not raised.')
import unittest
from holoviews.core import AARectangle, BoundingBox
def test_right_of(self):
    self.assertFalse(self.region.contains(1, 0))
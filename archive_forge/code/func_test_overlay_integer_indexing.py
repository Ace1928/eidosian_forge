import unittest
import numpy as np
from holoviews import Element, NdOverlay
def test_overlay_integer_indexing(self):
    overlay = NdOverlay(list(enumerate([self.view1, self.view2, self.view3])))
    self.assertEqual(overlay[0], self.view1)
    self.assertEqual(overlay[1], self.view2)
    self.assertEqual(overlay[2], self.view3)
    try:
        overlay[3]
        raise AssertionError('Index should be out of range.')
    except KeyError:
        pass
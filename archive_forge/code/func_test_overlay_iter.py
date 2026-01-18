import unittest
import numpy as np
from holoviews import Element, NdOverlay
def test_overlay_iter(self):
    views = [self.view1, self.view2, self.view3]
    overlay = NdOverlay(list(enumerate(views)))
    for el, v in zip(overlay, views):
        self.assertEqual(el, v)
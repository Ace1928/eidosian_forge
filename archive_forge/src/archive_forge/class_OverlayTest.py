import unittest
import numpy as np
from holoviews import Element, NdOverlay
class OverlayTest(CompositeTest):

    def test_overlay(self):
        NdOverlay(list(enumerate([self.view1, self.view2, self.view3])))

    def test_overlay_iter(self):
        views = [self.view1, self.view2, self.view3]
        overlay = NdOverlay(list(enumerate(views)))
        for el, v in zip(overlay, views):
            self.assertEqual(el, v)

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
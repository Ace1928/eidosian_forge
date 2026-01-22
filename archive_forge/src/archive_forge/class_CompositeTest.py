import unittest
import numpy as np
from holoviews import Element, NdOverlay
class CompositeTest(unittest.TestCase):
    """For testing of basic composite element types"""

    def setUp(self):
        self.data1 = np.zeros((10, 2))
        self.data2 = np.ones((10, 2))
        self.data3 = np.ones((10, 2)) * 2
        self.view1 = Element(self.data1, label='view1')
        self.view2 = Element(self.data2, label='view2')
        self.view3 = Element(self.data3, label='view3')
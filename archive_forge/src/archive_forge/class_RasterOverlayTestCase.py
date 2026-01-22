import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
class RasterOverlayTestCase(RasterTestCase):

    def setUp(self):
        super().setUp()
        self.overlay1_depth2 = self.mat1 * self.mat2
        self.overlay2_depth2 = self.mat1 * self.mat3
        self.overlay3_depth2 = self.mat4 * self.mat5
        self.overlay4_depth3 = self.mat1 * self.mat2 * self.mat3
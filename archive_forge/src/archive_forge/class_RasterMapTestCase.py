import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
class RasterMapTestCase(RasterOverlayTestCase):

    def setUp(self):
        super().setUp()
        self.map1_1D = HoloMap(kdims=['int'])
        self.map1_1D[0] = self.mat1
        self.map1_1D[1] = self.mat2
        self.map2_1D = HoloMap(kdims=['int'])
        self.map2_1D[1] = self.mat1
        self.map2_1D[2] = self.mat2
        self.map3_1D = HoloMap(kdims=['int'])
        self.map3_1D[1] = self.mat1
        self.map3_1D[2] = self.mat2
        self.map3_1D[3] = self.mat3
        self.map4_1D = HoloMap(kdims=['int'])
        self.map4_1D[0] = self.mat1
        self.map4_1D[1] = self.mat3
        self.map5_1D = HoloMap(kdims=['int'])
        self.map5_1D[0] = self.mat4
        self.map5_1D[1] = self.mat5
        self.map6_1D = HoloMap(kdims=['int_v2'])
        self.map6_1D[0] = self.mat1
        self.map6_1D[1] = self.mat2
        self.map7_1D = HoloMap(kdims=['int'])
        self.map7_1D[0] = self.overlay1_depth2
        self.map7_1D[1] = self.overlay2_depth2
        self.map8_1D = HoloMap(kdims=['int'])
        self.map8_1D[0] = self.overlay2_depth2
        self.map8_1D[1] = self.overlay1_depth2
        self.map1_2D = HoloMap(kdims=['int', Dimension('float')])
        self.map1_2D[0, 0.5] = self.mat1
        self.map1_2D[1, 1.0] = self.mat2
        self.map2_2D = HoloMap(kdims=['int', Dimension('float')])
        self.map2_2D[0, 1.0] = self.mat1
        self.map2_2D[1, 1.5] = self.mat2
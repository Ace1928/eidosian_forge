import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
class TestRGB(ComparisonTestCase):

    def setUp(self):
        self.rgb_array = np.random.randint(0, 255, (3, 3, 4))

    def test_construct_from_array_with_alpha(self):
        rgb = RGB(self.rgb_array)
        self.assertEqual(len(rgb.vdims), 4)

    def test_construct_from_tuple_with_alpha(self):
        rgb = RGB(([0, 1, 2], [0, 1, 2], self.rgb_array))
        self.assertEqual(len(rgb.vdims), 4)

    def test_construct_from_dict_with_alpha(self):
        rgb = RGB({'x': [1, 2, 3], 'y': [1, 2, 3], ('R', 'G', 'B', 'A'): self.rgb_array})
        self.assertEqual(len(rgb.vdims), 4)

    def test_not_using_class_variables_vdims(self):
        init_vdims = RGB(self.rgb_array).vdims
        cls_vdims = RGB.vdims
        for i, c in zip(init_vdims, cls_vdims):
            assert i is not c
            assert i == c

    def test_nodata(self):
        N = 2
        rgb_d = np.linspace(0, 1, N * N * 3).reshape(N, N, 3)
        rgb = RGB(rgb_d)
        assert sum(np.isnan(rgb['R'])) == 0
        assert sum(np.isnan(rgb['G'])) == 0
        assert sum(np.isnan(rgb['B'])) == 0
        rgb_n = rgb.redim.nodata(R=0)
        assert sum(np.isnan(rgb_n['R'])) == 1
        assert sum(np.isnan(rgb_n['G'])) == 0
        assert sum(np.isnan(rgb_n['B'])) == 0
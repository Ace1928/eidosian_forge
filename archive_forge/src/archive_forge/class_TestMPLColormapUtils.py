import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
class TestMPLColormapUtils(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.mpl

    def test_mpl_colormap_fire(self):
        colors = process_cmap('fire', 3, provider='matplotlib')
        self.assertEqual(colors, ['#000000', '#ed1400', '#ffffff'])

    def test_mpl_colormap_fire_r(self):
        colors = process_cmap('fire_r', 3, provider='matplotlib')
        self.assertEqual(colors, ['#ffffff', '#eb1300', '#000000'])

    def test_mpl_colormap_name_palette(self):
        colors = process_cmap('Greys', 3, provider='matplotlib')
        self.assertEqual(colors, ['#ffffff', '#959595', '#000000'])

    def test_mpl_colormap_instance(self):
        try:
            from matplotlib import colormaps
            cmap = colormaps.get('Greys')
        except ImportError:
            from matplotlib.cm import get_cmap
            cmap = get_cmap('Greys')
        colors = process_cmap(cmap, 3, provider='matplotlib')
        self.assertEqual(colors, ['#ffffff', '#959595', '#000000'])

    def test_mpl_colormap_categorical(self):
        colors = mplcmap_to_palette('Category20', 3)
        self.assertEqual(colors, ['#1f77b4', '#c5b0d5', '#9edae5'])

    def test_mpl_colormap_categorical_reverse(self):
        colors = mplcmap_to_palette('Category20_r', 3)
        self.assertEqual(colors, ['#1f77b4', '#8c564b', '#9edae5'][::-1])

    def test_mpl_colormap_sequential(self):
        colors = mplcmap_to_palette('YlGn', 3)
        self.assertEqual(colors, ['#ffffe5', '#77c578', '#004529'])

    def test_mpl_colormap_sequential_reverse(self):
        colors = mplcmap_to_palette('YlGn_r', 3)
        self.assertEqual(colors, ['#ffffe5', '#78c679', '#004529'][::-1])

    def test_mpl_colormap_diverging(self):
        colors = mplcmap_to_palette('RdBu', 3)
        self.assertEqual(colors, ['#67001f', '#f6f6f6', '#053061'])

    def test_mpl_colormap_diverging_reverse(self):
        colors = mplcmap_to_palette('RdBu_r', 3)
        self.assertEqual(colors, ['#67001f', '#f7f6f6', '#053061'][::-1])

    def test_mpl_colormap_perceptually_uniform(self):
        colors = mplcmap_to_palette('viridis', 4)
        self.assertEqual(colors, ['#440154', '#30678d', '#35b778', '#fde724'])

    def test_mpl_colormap_perceptually_uniform_reverse(self):
        colors = mplcmap_to_palette('viridis_r', 4)
        self.assertEqual(colors, ['#440154', '#30678d', '#35b778', '#fde724'][::-1])
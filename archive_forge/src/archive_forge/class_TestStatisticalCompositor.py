import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
@pytest.mark.usefixtures('mpl_backend')
class TestStatisticalCompositor:

    def setup_method(self):
        pytest.importorskip('scipy')

    def test_distribution_composite(self):
        dist = Distribution(np.array([0, 1, 2]))
        area = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(area, Area)
        assert area.vdims == [Dimension(('Value_density', 'Density'))]

    def test_distribution_composite_transfer_opts(self):
        dist = Distribution(np.array([0, 1, 2])).opts(color='red')
        area = Compositor.collapse_element(dist, backend='matplotlib')
        opts = Store.lookup_options('matplotlib', area, 'style').kwargs
        assert opts.get('color', None) == 'red'

    def test_distribution_composite_transfer_opts_with_group(self):
        dist = Distribution(np.array([0, 1, 2]), group='Test').opts(color='red')
        area = Compositor.collapse_element(dist, backend='matplotlib')
        opts = Store.lookup_options('matplotlib', area, 'style').kwargs
        assert opts.get('color', None) == 'red'

    def test_distribution_composite_custom_vdim(self):
        dist = Distribution(np.array([0, 1, 2]), vdims=['Test'])
        area = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(area, Area)
        assert area.vdims == [Dimension('Test')]

    def test_distribution_composite_not_filled(self):
        dist = Distribution(np.array([0, 1, 2])).opts(filled=False)
        curve = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(curve, Curve)
        assert curve.vdims == [Dimension(('Value_density', 'Density'))]

    def test_distribution_composite_empty_not_filled(self):
        dist = Distribution([]).opts(filled=False)
        curve = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(curve, Curve)
        assert curve.vdims == [Dimension(('Value_density', 'Density'))]

    def test_bivariate_composite(self):
        dist = Bivariate(np.random.rand(10, 2))
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Contours)
        assert contours.vdims == [Dimension('Density')]

    def test_bivariate_composite_transfer_opts(self):
        dist = Bivariate(np.random.rand(10, 2)).opts(cmap='Blues')
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        opts = Store.lookup_options('matplotlib', contours, 'style').kwargs
        assert opts.get('cmap', None) == 'Blues'

    def test_bivariate_composite_transfer_opts_with_group(self):
        dist = Bivariate(np.random.rand(10, 2), group='Test').opts(cmap='Blues')
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        opts = Store.lookup_options('matplotlib', contours, 'style').kwargs
        assert opts.get('cmap', None) == 'Blues'

    def test_bivariate_composite_custom_vdim(self):
        dist = Bivariate(np.random.rand(10, 2), vdims=['Test'])
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Contours)
        assert contours.vdims == [Dimension('Test')]

    def test_bivariate_composite_filled(self):
        dist = Bivariate(np.random.rand(10, 2)).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Polygons)
        assert contours.vdims[0].name == 'Density'

    def test_bivariate_composite_empty_filled(self):
        dist = Bivariate([]).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Polygons)
        assert contours.vdims == [Dimension('Density')]
        assert len(contours) == 0

    def test_bivariate_composite_empty_not_filled(self):
        dist = Bivariate([]).opts(filled=True)
        contours = Compositor.collapse_element(dist, backend='matplotlib')
        assert isinstance(contours, Contours)
        assert contours.vdims == [Dimension('Density')]
        assert len(contours) == 0
import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
class TestCrossBackendOptionSpecification(ComparisonTestCase):
    """
    Test the style system can style a single object across backends.
    """

    def setUp(self):
        self.plotly_options = Store._options.pop('plotly', None)
        self.store_mpl = OptionTree(sorted(Store.options(backend='matplotlib').items()), groups=Options._option_groups, backend='matplotlib')
        self.store_bokeh = OptionTree(sorted(Store.options(backend='bokeh').items()), groups=Options._option_groups, backend='bokeh')
        super().setUp()

    def tearDown(self):
        Store.options(val=self.store_mpl, backend='matplotlib')
        Store.options(val=self.store_bokeh, backend='bokeh')
        Store.current_backend = 'matplotlib'
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}
        if self.plotly_options is not None:
            Store._options['plotly'] = self.plotly_options
        super().tearDown()

    def assert_output_options_group_empty(self, obj):
        mpl_output_lookup = Store.lookup_options('matplotlib', obj, 'output').options
        self.assertEqual(mpl_output_lookup, {})
        bokeh_output_lookup = Store.lookup_options('bokeh', obj, 'output').options
        self.assertEqual(bokeh_output_lookup, {})

    def test_mpl_bokeh_mpl_via_option_objects_opts_method(self):
        img = Image(np.random.rand(10, 10))
        mpl_opts = Options('Image', cmap='Blues', backend='matplotlib')
        bokeh_opts = Options('Image', cmap='Purple', backend='bokeh')
        self.assertEqual(mpl_opts.kwargs['backend'], 'matplotlib')
        self.assertEqual(bokeh_opts.kwargs['backend'], 'bokeh')
        img.opts(mpl_opts, bokeh_opts)
        mpl_lookup = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_lookup['cmap'], 'Blues')
        bokeh_lookup = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_lookup['cmap'], 'Purple')
        self.assert_output_options_group_empty(img)

    def test_mpl_bokeh_mpl_via_builders_opts_method(self):
        img = Image(np.random.rand(10, 10))
        mpl_opts = opts.Image(cmap='Blues', backend='matplotlib')
        bokeh_opts = opts.Image(cmap='Purple', backend='bokeh')
        self.assertEqual(mpl_opts.kwargs['backend'], 'matplotlib')
        self.assertEqual(bokeh_opts.kwargs['backend'], 'bokeh')
        img.opts(mpl_opts, bokeh_opts)
        mpl_lookup = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_lookup['cmap'], 'Blues')
        bokeh_lookup = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_lookup['cmap'], 'Purple')
        self.assert_output_options_group_empty(img)

    def test_mpl_bokeh_mpl_via_dict_backend_keyword(self):
        curve = Curve([1, 2, 3])
        styled_mpl = curve.opts({'Curve': dict(color='red')}, backend='matplotlib')
        styled = styled_mpl.opts({'Curve': dict(color='green')}, backend='bokeh')
        mpl_lookup = Store.lookup_options('matplotlib', styled, 'style')
        self.assertEqual(mpl_lookup.kwargs['color'], 'red')
        bokeh_lookup = Store.lookup_options('bokeh', styled, 'style')
        self.assertEqual(bokeh_lookup.kwargs['color'], 'green')

    def test_mpl_bokeh_mpl_via_builders_opts_method_implicit_backend(self):
        img = Image(np.random.rand(10, 10))
        Store.set_current_backend('matplotlib')
        mpl_opts = opts.Image(cmap='Blues')
        bokeh_opts = opts.Image(cmap='Purple', backend='bokeh')
        self.assertEqual('backend' not in mpl_opts.kwargs, True)
        self.assertEqual(bokeh_opts.kwargs['backend'], 'bokeh')
        img.opts(mpl_opts, bokeh_opts)
        mpl_lookup = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_lookup['cmap'], 'Blues')
        bokeh_lookup = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_lookup['cmap'], 'Purple')
        self.assert_output_options_group_empty(img)

    def test_mpl_bokeh_mpl_via_builders_opts_method_literal_implicit_backend(self):
        img = Image(np.random.rand(10, 10))
        curve = Curve([1, 2, 3])
        overlay = img * curve
        Store.set_current_backend('matplotlib')
        literal = {'Curve': dict(color='orange'), 'Image': dict(cmap='jet', backend='bokeh')}
        styled = overlay.opts(literal)
        mpl_curve_lookup = Store.lookup_options('matplotlib', styled.Curve.I, 'style')
        self.assertEqual(mpl_curve_lookup.kwargs['color'], 'orange')
        mpl_img_lookup = Store.lookup_options('matplotlib', styled.Image.I, 'style')
        self.assertNotEqual(mpl_img_lookup.kwargs['cmap'], 'jet')
        bokeh_curve_lookup = Store.lookup_options('bokeh', styled.Curve.I, 'style')
        self.assertNotEqual(bokeh_curve_lookup.kwargs['color'], 'orange')
        bokeh_img_lookup = Store.lookup_options('bokeh', styled.Image.I, 'style')
        self.assertEqual(bokeh_img_lookup.kwargs['cmap'], 'jet')

    def test_mpl_bokeh_mpl_via_builders_opts_method_literal_explicit_backend(self):
        img = Image(np.random.rand(10, 10))
        curve = Curve([1, 2, 3])
        overlay = img * curve
        Store.set_current_backend('matplotlib')
        literal = {'Curve': dict(color='orange', backend='matplotlib'), 'Image': dict(cmap='jet', backend='bokeh')}
        styled = overlay.opts(literal)
        mpl_curve_lookup = Store.lookup_options('matplotlib', styled.Curve.I, 'style')
        self.assertEqual(mpl_curve_lookup.kwargs['color'], 'orange')
        mpl_img_lookup = Store.lookup_options('matplotlib', styled.Image.I, 'style')
        self.assertNotEqual(mpl_img_lookup.kwargs['cmap'], 'jet')
        bokeh_curve_lookup = Store.lookup_options('bokeh', styled.Curve.I, 'style')
        self.assertNotEqual(bokeh_curve_lookup.kwargs['color'], 'orange')
        bokeh_img_lookup = Store.lookup_options('bokeh', styled.Image.I, 'style')
        self.assertEqual(bokeh_img_lookup.kwargs['cmap'], 'jet')

    def test_mpl_bokeh_output_options_group_expandable(self):
        original_allowed_kws = Options._output_allowed_kws[:]
        Options._output_allowed_kws = ['backend', 'file_format_example']
        Store.register({Curve: plotting.mpl.CurvePlot}, 'matplotlib')
        Store.register({Curve: plotting.bokeh.CurvePlot}, 'bokeh')
        curve_bk = Options('Curve', backend='bokeh', color='blue')
        curve_mpl = Options('Curve', backend='matplotlib', color='red', file_format_example='SVG')
        c = Curve([1, 2, 3])
        styled = c.opts(curve_bk, curve_mpl)
        self.assertEqual(Store.lookup_options('matplotlib', styled, 'output').kwargs, {'backend': 'matplotlib', 'file_format_example': 'SVG'})
        self.assertEqual(Store.lookup_options('bokeh', styled, 'output').kwargs, {})
        Options._output_allowed_kws = original_allowed_kws
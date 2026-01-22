from unittest import skip, skipIf
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews.core.options import Cycle, Store
from holoviews.element import ErrorBars, Points, Rectangles, Table, VSpan
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.util import linear_gradient
from holoviews.selection import link_selections
from holoviews.streams import SelectionXY
class TestLinkSelectionsBokeh(TestLinkSelections):
    __test__ = True

    def setUp(self):
        import holoviews.plotting.bokeh
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def element_color(self, element):
        color = element.opts.get('style').kwargs['color']
        if isinstance(color, str):
            return color
        else:
            return list(color)

    @skip('Coloring Bokeh table not yet supported')
    def test_layout_selection_points_table(self):
        pass

    @skip('Bokeh ErrorBars selection not yet supported')
    def test_overlay_points_errorbars(self):
        pass

    @skip('Bokeh ErrorBars selection not yet supported')
    def test_overlay_points_errorbars_dynamic(self):
        pass

    def test_empty_layout(self):
        df = pd.DataFrame({'x': [1, 2], 'y': [1, 2], 'cat': ['A', 'B']})
        checkboxes = pn.widgets.CheckBoxGroup(options=['A', 'B'])

        def func(check):
            return hv.Scatter(df[df['cat'].isin(check)])
        ls = hv.link_selections.instance()
        a = ls(hv.DynamicMap(pn.bind(func, checkboxes)))
        b = ls(hv.DynamicMap(pn.bind(func, checkboxes)))
        hv.renderer('bokeh').get_plot(a + b)
        checkboxes.value = ['A']
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
def test_points_selection(self, dynamic=False, show_regions=True):
    points = Points(self.data)
    if dynamic:
        points = hv.util.Dynamic(points)
    lnk_sel = link_selections.instance(show_regions=show_regions, unselected_color='#ff0000')
    linked = lnk_sel(points)
    current_obj = linked[()]
    self.assertIsInstance(current_obj, hv.Overlay)
    unselected, selected, region, region2 = current_obj.values()
    self.check_base_points_like(unselected, lnk_sel)
    self.check_overlay_points_like(selected, lnk_sel, self.data)
    selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
    self.assertIsInstance(selectionxy, hv.streams.SelectionXY)
    selectionxy.event(bounds=(0, 1, 5, 5))
    unselected, selected, region, region2 = linked[()].values()
    self.check_base_points_like(unselected, lnk_sel)
    self.check_overlay_points_like(selected, lnk_sel, self.data.iloc[1:])
    if show_regions:
        self.assertEqual(region, Rectangles([(0, 1, 5, 5)]))
    else:
        self.assertEqual(region, Rectangles([]))
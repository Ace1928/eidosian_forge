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
def test_points_selection_streaming(self):
    buffer = hv.streams.Buffer(self.data.iloc[:2], index=False)
    points = hv.DynamicMap(Points, streams=[buffer])
    lnk_sel = link_selections.instance(unselected_color='#ff0000')
    linked = lnk_sel(points)
    selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
    self.assertIsInstance(selectionxy, hv.streams.SelectionXY)
    selectionxy.event(bounds=(0, 0, 4, 2))
    current_obj = linked[()]
    self.check_base_points_like(current_obj.Points.I, lnk_sel, self.data.iloc[:2])
    self.check_overlay_points_like(current_obj.Points.II, lnk_sel, self.data.iloc[[0]])
    buffer.send(self.data.iloc[[2]])
    current_obj = linked[()]
    self.check_base_points_like(current_obj.Points.I, lnk_sel, self.data)
    self.check_overlay_points_like(current_obj.Points.II, lnk_sel, self.data.iloc[[0, 2]])
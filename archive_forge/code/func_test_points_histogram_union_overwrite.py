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
def test_points_histogram_union_overwrite(self, dynamic=False):
    self.do_crossfilter_points_histogram(selection_mode='union', cross_filter_mode='overwrite', selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2], points_region1=[(1, 1, 4, 4)], points_region2=[], points_region3=[(0, 0, 4, 2.5)], points_region4=[], hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2], dynamic=dynamic)
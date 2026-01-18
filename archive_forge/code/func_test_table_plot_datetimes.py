from datetime import datetime as dt
from bokeh.models.widgets import (
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import CDSCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import CDSStream, Stream
def test_table_plot_datetimes(self):
    table = Table([dt.now(), dt.now()], 'Date')
    plot = bokeh_renderer.get_plot(table)
    column = plot.state.columns[0]
    self.assertEqual(column.title, 'Date')
    self.assertIsInstance(column.formatter, DateFormatter)
    self.assertIsInstance(column.editor, DateEditor)
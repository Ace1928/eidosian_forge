import math
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.barcharts import BarChartProperties
from reportlab.graphics.widgetbase import TypedPropertyCollection
from Bio.Graphics import _write
def _determine_position(self, start_x, start_y, end_x, end_y):
    """Calculate the position of the chart with blank space (PRIVATE).

        This uses some padding around the chart, and takes into account
        whether the chart has a title. It returns 4 values, which are,
        in order, the x_start, x_end, y_start and y_end of the chart
        itself.
        """
    x_padding = self.padding_percent * (end_x - start_x)
    y_padding = self.padding_percent * (start_y - end_y)
    new_x_start = start_x + x_padding
    new_x_end = end_x - x_padding
    if self.chart_title:
        new_y_start = start_y - y_padding - self.chart_title_size
    else:
        new_y_start = start_y - y_padding
    new_y_end = end_y + y_padding
    return (new_x_start, new_x_end, new_y_start, new_y_end)
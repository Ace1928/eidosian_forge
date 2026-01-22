import math
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.barcharts import BarChartProperties
from reportlab.graphics.widgetbase import TypedPropertyCollection
from Bio.Graphics import _write
class BarChartDistribution:
    """Display the distribution of values as a bunch of bars."""

    def __init__(self, display_info=None):
        """Initialize a Bar Chart display of distribution info.

        Attributes:
         - display_info - the information to be displayed in the distribution.
           This should be ordered as a list of lists, where each internal list
           is a data set to display in the bar chart.

        """
        if display_info is None:
            display_info = []
        self.display_info = display_info
        self.x_axis_title = ''
        self.y_axis_title = ''
        self.chart_title = ''
        self.chart_title_size = 10
        self.padding_percent = 0.15

    def draw(self, cur_drawing, start_x, start_y, end_x, end_y):
        """Draw a bar chart with the info in the specified range."""
        bar_chart = VerticalBarChart()
        if self.chart_title:
            self._draw_title(cur_drawing, self.chart_title, start_x, start_y, end_x, end_y)
        x_start, x_end, y_start, y_end = self._determine_position(start_x, start_y, end_x, end_y)
        bar_chart.x = x_start
        bar_chart.y = y_start
        bar_chart.width = abs(x_start - x_end)
        bar_chart.height = abs(y_start - y_end)
        bar_chart.data = self.display_info
        bar_chart.valueAxis.valueMin = min(self.display_info[0])
        bar_chart.valueAxis.valueMax = max(self.display_info[0])
        for data_set in self.display_info[1:]:
            if min(data_set) < bar_chart.valueAxis.valueMin:
                bar_chart.valueAxis.valueMin = min(data_set)
            if max(data_set) > bar_chart.valueAxis.valueMax:
                bar_chart.valueAxis.valueMax = max(data_set)
        if len(self.display_info) == 1:
            bar_chart.groupSpacing = 0
            style = TypedPropertyCollection(BarChartProperties)
            style.strokeWidth = 0
            style.strokeColor = colors.green
            style[0].fillColor = colors.green
            bar_chart.bars = style
        cur_drawing.add(bar_chart)

    def _draw_title(self, cur_drawing, title, start_x, start_y, end_x, end_y):
        """Add the title of the figure to the drawing (PRIVATE)."""
        x_center = start_x + (end_x - start_x) / 2
        y_pos = end_y + self.padding_percent * (start_y - end_y) / 2
        title_string = String(x_center, y_pos, title)
        title_string.fontName = 'Helvetica-Bold'
        title_string.fontSize = self.chart_title_size
        title_string.textAnchor = 'middle'
        cur_drawing.add(title_string)

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
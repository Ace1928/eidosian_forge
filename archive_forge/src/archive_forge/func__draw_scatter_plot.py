from reportlab.lib import colors
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.markers import makeEmptySquare, makeFilledSquare
from reportlab.graphics.charts.markers import makeFilledDiamond, makeSmiley
from reportlab.graphics.charts.markers import makeFilledCircle, makeEmptyCircle
from Bio.Graphics import _write
def _draw_scatter_plot(self, cur_drawing, x_start, y_start, x_end, y_end):
    """Draw a scatter plot on the drawing with the given coordinates (PRIVATE)."""
    scatter_plot = LinePlot()
    scatter_plot.x = x_start
    scatter_plot.y = y_start
    scatter_plot.width = abs(x_start - x_end)
    scatter_plot.height = abs(y_start - y_end)
    scatter_plot.data = self.display_info
    scatter_plot.joinedLines = 0
    x_min, x_max, y_min, y_max = self._find_min_max(self.display_info)
    scatter_plot.xValueAxis.valueMin = x_min
    scatter_plot.xValueAxis.valueMax = x_max
    scatter_plot.xValueAxis.valueStep = (x_max - x_min) / 10.0
    scatter_plot.yValueAxis.valueMin = y_min
    scatter_plot.yValueAxis.valueMax = y_max
    scatter_plot.yValueAxis.valueStep = (y_max - y_min) / 10.0
    self._set_colors_and_shapes(scatter_plot, self.display_info)
    cur_drawing.add(scatter_plot)
from reportlab.lib import colors
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.markers import makeEmptySquare, makeFilledSquare
from reportlab.graphics.charts.markers import makeFilledDiamond, makeSmiley
from reportlab.graphics.charts.markers import makeFilledCircle, makeEmptyCircle
from Bio.Graphics import _write
def _set_colors_and_shapes(self, scatter_plot, display_info):
    """Set the colors and shapes of the points displayed (PRIVATE).

        By default this just sets all of the points according to the order
        of colors and shapes defined in self.color_choices and
        self.shape_choices. The first 5 shapes and colors are unique, the
        rest of them are just set to the same color and shape (since I
        ran out of shapes!).

        You can change how this function works by either changing the
        values of the color_choices and shape_choices attributes, or
        by inheriting from this class and overriding this function.
        """
    for value_num in range(len(display_info)):
        if value_num + 1 < len(self.color_choices):
            scatter_plot.lines[value_num].strokeColor = self.color_choices[value_num]
            scatter_plot.lines[value_num].symbol = self.shape_choices[value_num]
        else:
            scatter_plot.lines[value_num].strokeColor = self.color_choices[-1]
            scatter_plot.lines[value_num].symbol = self.shape_choices[-1]
import math
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.barcharts import BarChartProperties
from reportlab.graphics.widgetbase import TypedPropertyCollection
from Bio.Graphics import _write
class LineDistribution:
    """Display the distribution of values as connected lines.

    This distribution displays the change in values across the object as
    lines. This also allows multiple distributions to be displayed on a
    single graph.
    """

    def __init__(self):
        """Initialize the class."""

    def draw(self, cur_drawing, start_x, start_y, end_x, end_y):
        """Draw a line distribution into the current drawing."""
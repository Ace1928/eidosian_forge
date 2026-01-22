from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PlotTypeValueValuesEnum(_messages.Enum):
    """How this data should be plotted on the chart.

    Values:
      PLOT_TYPE_UNSPECIFIED: Plot type is unspecified. The view will default
        to LINE.
      LINE: The data is plotted as a set of lines (one line per series).
      STACKED_AREA: The data is plotted as a set of filled areas (one area per
        series), with the areas stacked vertically (the base of each area is
        the top of its predecessor, and the base of the first area is the
        x-axis). Since the areas do not overlap, each is filled with a
        different opaque color.
      STACKED_BAR: The data is plotted as a set of rectangular boxes (one box
        per series), with the boxes stacked vertically (the base of each box
        is the top of its predecessor, and the base of the first box is the
        x-axis). Since the boxes do not overlap, each is filled with a
        different opaque color.
      HEATMAP: The data is plotted as a heatmap. The series being plotted must
        have a DISTRIBUTION value type. The value of each bucket in the
        distribution is displayed as a color. This type is not currently
        available in the Stackdriver Monitoring application.
    """
    PLOT_TYPE_UNSPECIFIED = 0
    LINE = 1
    STACKED_AREA = 2
    STACKED_BAR = 3
    HEATMAP = 4
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChartTypeValueValuesEnum(_messages.Enum):
    """Required. Indicates the visualization type for the PieChart.

    Values:
      PIE_CHART_TYPE_UNSPECIFIED: The zero value. No type specified. Do not
        use.
      PIE: A Pie type PieChart.
      DONUT: Similar to PIE, but the DONUT type PieChart has a hole in the
        middle.
    """
    PIE_CHART_TYPE_UNSPECIFIED = 0
    PIE = 1
    DONUT = 2
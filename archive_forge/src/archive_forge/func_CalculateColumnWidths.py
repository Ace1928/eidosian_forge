from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def CalculateColumnWidths(self, max_column_width=None, indent_length=0):
    """See _Marker base class.

    Args:
      max_column_width: The maximum column width to allow. Overriden by the
        instance's max_column_width, if the instance has a max_column_width
        specified.
      indent_length: See _Marker base class.

    Returns:
      An empty ColumnWidths object.
    """
    effective_max_column_width = self._max_column_width or max_column_width
    self._column_widths = self._lines.CalculateColumnWidths(effective_max_column_width, indent_length)
    return ColumnWidths()
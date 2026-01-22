from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
@six.add_metaclass(abc.ABCMeta)
class CustomPrinterBase(resource_printer_base.ResourcePrinter):
    """Base to extend to custom-format a resource.

  Instead of using a format string, uses the "Transform" method to build a
  structure of marker classes that represent how to print out the resource
  in a structured way, and then prints it out in that way.

  A string prints out as a string; the marker classes above print out as an
  indented aligned table.
  """
    MAX_COLUMN_WIDTH = 20

    def __init__(self, *args, **kwargs):
        kwargs['process_record'] = self.Transform
        super(CustomPrinterBase, self).__init__(*args, **kwargs)

    def _AddRecord(self, record, delimit=True):
        if isinstance(record, _Marker):
            column_widths = record.CalculateColumnWidths(self.MAX_COLUMN_WIDTH)
            record.Print(self._out, 0, column_widths)
        elif record:
            self._out.write(_GenerateLineValue(record))
        if delimit:
            self._out.write('------\n')

    @abc.abstractmethod
    def Transform(self, record):
        """Override to describe the format of the record.

    Takes in the raw record, returns a structure of "marker classes" (above in
    this file) that will describe how to print it.

    Args:
      record: The record to transform
    Returns:
      A structure of "marker classes" that describes how to print the record.
    """
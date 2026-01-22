from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_property
def Printer(self, *args, **kwargs):
    """Calls the resource_printer.Printer() method (for nested printers)."""
    return self._printer(*args, **kwargs)
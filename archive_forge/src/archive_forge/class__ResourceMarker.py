from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_property
class _ResourceMarker(object):
    """A marker that can be injected into resource lists."""

    def Act(self, printer):
        """Called by ResourcePrinter.Addrecord().

    Args:
      printer: The printer object.
    """
        pass
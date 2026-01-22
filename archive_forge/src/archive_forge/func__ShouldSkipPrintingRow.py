from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _ShouldSkipPrintingRow(self, row):
    """Returns true if the given row should not be printed."""
    followed_by_empty = _FollowedByEmpty(row, 0) or _FollowedByMarkerWithNoOutput(row, 0)
    return not row or (self.skip_empty and followed_by_empty)
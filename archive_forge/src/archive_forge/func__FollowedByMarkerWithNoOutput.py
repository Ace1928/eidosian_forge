from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _FollowedByMarkerWithNoOutput(row, index):
    """Returns true if the column after the given index is a no-output _Marker."""
    next_index = index + 1
    return len(row) > next_index and isinstance(row[next_index], _Marker) and (not row[next_index].WillPrintOutput())
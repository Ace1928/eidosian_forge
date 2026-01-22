from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _MergeColumnWidths(self, other):
    """Merges another ColumnWidths into this instance."""
    for i, width in enumerate(other.widths):
        self._SetWidth(i, width)
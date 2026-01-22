from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_property
def RemoveHiddenColumns(self, row):
    """Returns a list of visible columns given a row."""
    if not self._is_column_visible:
        self._SetVisibleColumns()
    if self._is_column_visible:
        return [col for i, col in enumerate(row) if self._is_column_visible[i]]
    else:
        return row
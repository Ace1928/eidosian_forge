from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _set_source_and_destination_schemes(self, status_message):
    """Sets source and destination schemes, if available.

    Args:
      status_message (thread_messages.*): Message to process.
    """
    if self._source_scheme == UNSET:
        self._source_scheme = self._get_scheme_value(status_message.source_url)
    if self._destination_scheme == UNSET:
        self._destination_scheme = self._get_scheme_value(status_message.destination_url)
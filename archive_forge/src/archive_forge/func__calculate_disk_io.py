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
def _calculate_disk_io(self):
    """Calculate deltas of time spent on I/O."""
    current_os = platforms.OperatingSystem.Current()
    if current_os == platforms.OperatingSystem.LINUX:
        disk_start = self._disk_counters_start
        disk_end = get_disk_counters()
        return sum([stat[4] + stat[5] for stat in disk_end.values()]) - sum([stat[4] + stat[5] for stat in disk_start.values()])
    return UNSET
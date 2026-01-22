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
class MetricsReporter:
    """Mix-in for tracking metrics during task status reporting."""

    def __init__(self):
        self._source_scheme = UNSET
        self._destination_scheme = UNSET
        self._disk_counters_start = get_disk_counters()

    def _get_scheme_value(self, url):
        """Extracts the scheme as an integer value from a storage_url."""
        if url:
            return PROVIDER_PREFIX_TO_METRICS_KEY[url.scheme]
        return UNSET

    def _set_source_and_destination_schemes(self, status_message):
        """Sets source and destination schemes, if available.

    Args:
      status_message (thread_messages.*): Message to process.
    """
        if self._source_scheme == UNSET:
            self._source_scheme = self._get_scheme_value(status_message.source_url)
        if self._destination_scheme == UNSET:
            self._destination_scheme = self._get_scheme_value(status_message.destination_url)

    def _calculate_disk_io(self):
        """Calculate deltas of time spent on I/O."""
        current_os = platforms.OperatingSystem.Current()
        if current_os == platforms.OperatingSystem.LINUX:
            disk_start = self._disk_counters_start
            disk_end = get_disk_counters()
            return sum([stat[4] + stat[5] for stat in disk_end.values()]) - sum([stat[4] + stat[5] for stat in disk_start.values()])
        return UNSET

    def _report_metrics(self, total_bytes, time_delta, num_files):
        """Reports back all tracked events via report method.

    Args:
      total_bytes (int): Amount of data transferred in bytes.
      time_delta (int): Time elapsed during the transfer in seconds.
      num_files (int): Number of files processed
    """
        avg_speed = round(float(total_bytes) / float(time_delta))
        report(source_scheme=self._source_scheme, destination_scheme=self._destination_scheme, num_files=num_files, size=total_bytes, avg_speed=avg_speed, disk_io_time=self._calculate_disk_io())
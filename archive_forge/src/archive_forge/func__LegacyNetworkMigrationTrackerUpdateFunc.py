from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.networks import flags
from googlecloudsdk.command_lib.compute.networks import network_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def _LegacyNetworkMigrationTrackerUpdateFunc(self, tracker, operation, unused_status):
    latest_status_message = operation.statusMessage
    self._MarkStagesCompleted(tracker, latest_status_message)
    tracker.StartStage(latest_status_message)
    tracker.last_status_message = latest_status_message
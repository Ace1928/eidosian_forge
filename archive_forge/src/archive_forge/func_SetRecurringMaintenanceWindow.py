from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def SetRecurringMaintenanceWindow(self, cluster_ref, existing_policy, window_start, window_end, window_recurrence):
    """Sets a recurring maintenance window as the maintenance policy for a cluster.

    Args:
      cluster_ref: The cluster to update.
      existing_policy: The existing maintenance policy, if any.
      window_start: Start time of the window as a datetime.datetime.
      window_end: End time of the window as a datetime.datetime.
      window_recurrence: RRULE str defining how the window will recur.

    Returns:
      The operation from this cluster update.
    """
    recurring_window = self.messages.RecurringTimeWindow(window=self.messages.TimeWindow(startTime=window_start.isoformat(), endTime=window_end.isoformat()), recurrence=window_recurrence)
    if existing_policy is None:
        existing_policy = self.messages.MaintenancePolicy()
    if existing_policy.window is None:
        existing_policy.window = self.messages.MaintenanceWindow()
    existing_policy.window.dailyMaintenanceWindow = None
    existing_policy.window.recurringWindow = recurring_window
    return self._SendMaintenancePolicyRequest(cluster_ref, existing_policy)
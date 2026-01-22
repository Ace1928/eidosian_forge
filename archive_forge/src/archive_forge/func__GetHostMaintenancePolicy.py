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
def _GetHostMaintenancePolicy(options, messages):
    """Get HostMaintenancePolicy from options."""
    if options.host_maintenance_interval is not None:
        maintenance_interval_types = {'UNSPECIFIED': messages.HostMaintenancePolicy.MaintenanceIntervalValueValuesEnum.MAINTENANCE_INTERVAL_UNSPECIFIED, 'PERIODIC': messages.HostMaintenancePolicy.MaintenanceIntervalValueValuesEnum.PERIODIC, 'AS_NEEDED': messages.HostMaintenancePolicy.MaintenanceIntervalValueValuesEnum.AS_NEEDED}
        if options.host_maintenance_interval not in maintenance_interval_types:
            raise util.Error(HOST_MAINTENANCE_INTERVAL_TYPE_NOT_SUPPORTED.FORMAT(type=options.host_maintenance_interval))
        return messages.HostMaintenancePolicy(maintenanceInterval=maintenance_interval_types[options.host_maintenance_interval])
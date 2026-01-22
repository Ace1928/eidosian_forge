from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def CreateSchedulingMessage(messages, maintenance_policy, preemptible, restart_on_failure, node_affinities=None, min_node_cpu=None, location_hint=None, maintenance_freeze_duration=None, maintenance_interval=None, provisioning_model=None, instance_termination_action=None, host_error_timeout_seconds=None, max_run_duration=None, termination_time=None, local_ssd_recovery_timeout=None, availability_domain=None, graceful_shutdown=None, discard_local_ssds_at_termination_timestamp=None):
    """Create scheduling message for VM."""
    on_host_maintenance = CreateOnHostMaintenanceMessage(messages, maintenance_policy)
    if preemptible or provisioning_model == 'SPOT':
        scheduling = messages.Scheduling(automaticRestart=False, onHostMaintenance=on_host_maintenance, preemptible=True)
    else:
        scheduling = messages.Scheduling(automaticRestart=restart_on_failure, onHostMaintenance=on_host_maintenance)
    if provisioning_model:
        scheduling.provisioningModel = messages.Scheduling.ProvisioningModelValueValuesEnum(provisioning_model)
    if instance_termination_action:
        scheduling.instanceTerminationAction = messages.Scheduling.InstanceTerminationActionValueValuesEnum(instance_termination_action)
    if max_run_duration is not None:
        scheduling.maxRunDuration = messages.Duration(seconds=max_run_duration)
    if local_ssd_recovery_timeout is not None:
        scheduling.localSsdRecoveryTimeout = messages.Duration(seconds=local_ssd_recovery_timeout)
    if graceful_shutdown is not None:
        scheduling.gracefulShutdown = messages.SchedulingGracefulShutdown()
        if 'enabled' in graceful_shutdown:
            scheduling.gracefulShutdown.enabled = graceful_shutdown['enabled']
        if 'maxDuration' in graceful_shutdown:
            scheduling.gracefulShutdown.maxDuration = messages.Duration(seconds=graceful_shutdown['maxDuration'])
    if termination_time:
        scheduling.terminationTime = times.FormatDateTime(termination_time)
    if node_affinities:
        scheduling.nodeAffinities = node_affinities
    if min_node_cpu is not None:
        scheduling.minNodeCpus = int(min_node_cpu)
    if location_hint:
        scheduling.locationHint = location_hint
    if maintenance_freeze_duration:
        scheduling.maintenanceFreezeDurationHours = maintenance_freeze_duration // 3600
    if maintenance_interval:
        scheduling.maintenanceInterval = messages.Scheduling.MaintenanceIntervalValueValuesEnum(maintenance_interval)
    if host_error_timeout_seconds:
        scheduling.hostErrorTimeoutSeconds = host_error_timeout_seconds
    if availability_domain:
        scheduling.availabilityDomain = availability_domain
    if discard_local_ssds_at_termination_timestamp:
        scheduling.onInstanceStopAction = messages.SchedulingOnInstanceStopAction(discardLocalSsd=discard_local_ssds_at_termination_timestamp)
    return scheduling
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
def CreateAdvancedMachineFeaturesMessage(messages, enable_nested_virtualization=None, threads_per_core=None, numa_node_count=None, visible_core_count=None, enable_uefi_networking=None, performance_monitoring_unit=None, enable_watchdog_timer=None):
    """Create AdvancedMachineFeatures message for an Instance."""
    features = messages.AdvancedMachineFeatures()
    if enable_nested_virtualization is not None:
        features.enableNestedVirtualization = enable_nested_virtualization
    if threads_per_core is not None:
        features.threadsPerCore = threads_per_core
    if numa_node_count is not None:
        features.numaNodeCount = numa_node_count
    if visible_core_count is not None:
        features.visibleCoreCount = visible_core_count
    if enable_uefi_networking is not None:
        features.enableUefiNetworking = enable_uefi_networking
    if performance_monitoring_unit is not None:
        features.performanceMonitoringUnit = messages.AdvancedMachineFeatures.PerformanceMonitoringUnitValueValuesEnum(performance_monitoring_unit.upper())
    if enable_watchdog_timer is not None:
        features.enableWatchdogTimer = enable_watchdog_timer
    return features
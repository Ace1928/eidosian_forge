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
def GetNetworkPerformanceConfig(args, client):
    """Get NetworkPerformanceConfig message for the instance."""
    network_perf_args = getattr(args, 'network_performance_configs', [])
    network_perf_configs = client.messages.NetworkPerformanceConfig()
    for config in network_perf_args:
        total_tier = config.get('total-egress-bandwidth-tier', '').upper()
        if total_tier:
            network_perf_configs.totalEgressBandwidthTier = client.messages.NetworkPerformanceConfig.TotalEgressBandwidthTierValueValuesEnum(total_tier)
    return network_perf_configs
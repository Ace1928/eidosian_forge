from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddNetworkPerformanceConfigsArgs(parser):
    """Adds config flags for advanced networking bandwidth tiers."""
    network_perf_config_help = '      Configures network performance settings for the instance.\n      If this flag is not specified, the instance will be created\n      with its default network performance configuration.\n\n      *total-egress-bandwidth-tier*::: Total egress bandwidth is the available\n      outbound bandwidth from a VM, regardless of whether the traffic\n      is going to internal IP or external IP destinations.\n      The following tier values are allowed: [{tier_values}]\n\n      '.format(tier_values=','.join([six.text_type(tier_val) for tier_val in constants.ADV_NETWORK_TIER_CHOICES]))
    spec = {'total-egress-bandwidth-tier': str}
    parser.add_argument('--network-performance-configs', type=arg_parsers.ArgDict(spec=spec), action='append', metavar='PROPERTY=VALUE', help=network_perf_config_help)
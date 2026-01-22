from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def ValidateMigStatefulIPFlagsForInstanceConfigs(args, current_internal_addresses, current_external_addresses, for_update=False):
    """Validates the values of stateful flags for instance configs, with IPs."""
    ValidateMigStatefulIpFlagForInstanceConfigs('--stateful-internal-ip', args.stateful_internal_ip, current_internal_addresses)
    ValidateMigStatefulIpFlagForInstanceConfigs('--stateful-external-ip', args.stateful_external_ip, current_external_addresses)
    if for_update:
        ValidateMigStatefulIpsRemovalFlagForInstanceConfigs(flag_name='--remove-stateful-internal-ips', ips_to_remove=args.remove_stateful_internal_ips, ips_to_update=args.stateful_internal_ip)
        ValidateMigStatefulIpsRemovalFlagForInstanceConfigs(flag_name='--remove-stateful-external-ips', ips_to_remove=args.remove_stateful_external_ips, ips_to_update=args.stateful_external_ip)
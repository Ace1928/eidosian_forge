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
def AddMigStatefulIPsFlagsForUpdateInstanceConfigs(parser):
    """Add args for per-instance configs update command."""
    ip_help_text = textwrap.dedent(STATEFUL_IPS_HELP_INSTANCE_CONFIGS_UPDATE + STATEFUL_IP_INTERFACE_NAME_ARG_WITH_ADDRESS_HELP + STATEFUL_IP_ADDRESS_ARG_OPTIONAL_HELP + STATEFUL_IP_AUTO_DELETE_ARG_HELP)
    remove_ip_help_text = "\n      List of all stateful IP network interface names to remove from\n      the instance's per-instance configuration.\n      "
    _AddMigStatefulIPsFlags(parser, '--stateful-internal-ip', ip_help_text, '--remove-stateful-internal-ips', remove_ip_help_text)
    _AddMigStatefulIPsFlags(parser, '--stateful-external-ip', ip_help_text, '--remove-stateful-external-ips', remove_ip_help_text)
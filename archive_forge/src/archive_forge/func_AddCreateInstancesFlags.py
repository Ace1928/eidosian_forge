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
def AddCreateInstancesFlags(parser):
    """Adding stateful flags for creating and updating instance configs."""
    parser.add_argument('--instance', required=True, help='Name of the new instance to create.')
    parser.add_argument('--stateful-disk', type=arg_parsers.ArgDict(spec={'device-name': str, 'source': str, 'mode': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName('--stateful-disk')}), action='append', help=textwrap.dedent(STATEFUL_DISKS_HELP_INSTANCE_CONFIGS))
    stateful_metadata_argument_name = '--stateful-metadata'
    parser.add_argument(stateful_metadata_argument_name, type=arg_parsers.ArgDict(min_length=1), default={}, action=arg_parsers.StoreOnceAction, metavar='KEY=VALUE', help=textwrap.dedent(STATEFUL_METADATA_HELP.format(argument_name=stateful_metadata_argument_name)))
    stateful_ips_help_text_template = textwrap.dedent(STATEFUL_IPS_HELP_BASE + STATEFUL_IPS_HELP_TEMPLATE + STATEFUL_IP_INTERFACE_NAME_ARG_WITH_ADDRESS_HELP + STATEFUL_IP_ADDRESS_ARG_HELP + STATEFUL_IP_AUTO_DELETE_ARG_HELP)
    stateful_internal_ip_flag_name = '--stateful-internal-ip'
    parser.add_argument(stateful_internal_ip_flag_name, type=arg_parsers.ArgDict(spec={'interface-name': str, 'address': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(stateful_internal_ip_flag_name)}), action='append', help=stateful_ips_help_text_template.format(ip_type='internal'))
    stateful_external_ip_flag_name = '--stateful-external-ip'
    parser.add_argument(stateful_external_ip_flag_name, type=arg_parsers.ArgDict(spec={'interface-name': str, 'address': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(stateful_external_ip_flag_name)}), action='append', help=stateful_ips_help_text_template.format(ip_type='external'))
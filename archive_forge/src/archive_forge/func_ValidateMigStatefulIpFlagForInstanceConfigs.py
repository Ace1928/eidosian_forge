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
def ValidateMigStatefulIpFlagForInstanceConfigs(flag_name, stateful_ips, current_addresses):
    """Validates the values of stateful IP flags for instance configs."""
    interface_names = set()
    for stateful_ip in stateful_ips or []:
        interface_name = stateful_ip.get('interface-name', STATEFUL_IP_DEFAULT_INTERFACE_NAME)
        if not ('address' in stateful_ip or interface_name in current_addresses):
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message='[address] is required')
        if interface_name in interface_names:
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message='[interface-name] `{0}` is not unique in the collection'.format(interface_name))
        interface_names.add(interface_name)
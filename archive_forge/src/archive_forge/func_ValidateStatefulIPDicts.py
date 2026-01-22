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
def ValidateStatefulIPDicts(stateful_ips, flag_name):
    """Validate enabled, interface-name and auto-delete flags in a stateful IP."""
    interface_names = set()
    for stateful_ip in stateful_ips or []:
        if not (stateful_ip.get('interface-name') or 'enabled' in stateful_ip):
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message='one of: [interface-name], [enabled] is required.')
        interface_name = stateful_ip.get('interface-name', STATEFUL_IP_DEFAULT_INTERFACE_NAME)
        if interface_name in interface_names:
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message='[interface-name] `{0}` is not unique in the collection'.format(interface_name))
        interface_names.add(interface_name)
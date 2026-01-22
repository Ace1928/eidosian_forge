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
def ValidateMigStatefulIpsRemovalFlagForInstanceConfigs(flag_name, ips_to_remove, ips_to_update):
    remove_ips_set = set(ips_to_remove or [])
    for ip_to_update in ips_to_update or []:
        if ip_to_update.get('interface-name') in remove_ips_set:
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message='the same [interface-name] `{0}` cannot be updated and removed in one command call'.format(ip_to_update.get('interface-name')))
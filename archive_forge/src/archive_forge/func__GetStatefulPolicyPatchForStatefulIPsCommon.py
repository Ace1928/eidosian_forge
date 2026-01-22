from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import stateful_policy_utils as policy_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def _GetStatefulPolicyPatchForStatefulIPsCommon(self, client, update_ip_to_ip_entry_lambda, update_ip_to_none_lambda, update_ips=None, remove_interface_names=None):
    if remove_interface_names:
        managed_instance_groups_utils.RegisterCustomStatefulIpsPatchEncoders(client)
    patched_ips_map = {}
    for update_ip in update_ips or []:
        interface_name = update_ip.get('interface-name', instance_groups_flags.STATEFUL_IP_DEFAULT_INTERFACE_NAME)
        updated_preserved_state_ip = update_ip_to_ip_entry_lambda(update_ip)
        patched_ips_map[interface_name] = updated_preserved_state_ip
    for interface_name in remove_interface_names or []:
        updated_preserved_state_ip = update_ip_to_none_lambda(interface_name)
        patched_ips_map[interface_name] = updated_preserved_state_ip
    stateful_ips = sorted([stateful_ip for key, stateful_ip in six.iteritems(patched_ips_map)], key=lambda x: x.key)
    return stateful_ips
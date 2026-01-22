from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_getter
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_messages
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
@staticmethod
def _CreateInstanceReference(holder, igm_ref, instance_name):
    """Creates reference to instance in instance group (zonal or regional)."""
    if instance_name.startswith('https://') or instance_name.startswith('http://'):
        return holder.resources.ParseURL(instance_name)
    instance_references = managed_instance_groups_utils.CreateInstanceReferences(holder=holder, igm_ref=igm_ref, instance_names=[instance_name])
    if not instance_references:
        raise managed_instance_groups_utils.ResourceCannotBeResolvedException('Instance name {0} cannot be resolved'.format(instance_name))
    return instance_references[0]
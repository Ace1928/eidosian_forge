from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import partner_metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import flags as sole_tenancy_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.util.args import labels_util
def _GetShieldedInstanceConfigRef(self, instance_ref, args, holder):
    client = holder.client.apitools_client
    messages = holder.client.messages
    if args.shielded_vm_secure_boot is None and args.shielded_vm_vtpm is None and (args.shielded_vm_integrity_monitoring is None):
        return None
    shieldedinstance_config_message = instance_utils.CreateShieldedInstanceConfigMessage(messages, args.shielded_vm_secure_boot, args.shielded_vm_vtpm, args.shielded_vm_integrity_monitoring)
    request = messages.ComputeInstancesUpdateShieldedInstanceConfigRequest(instance=instance_ref.Name(), project=instance_ref.project, shieldedInstanceConfig=shieldedinstance_config_message, zone=instance_ref.zone)
    operation = client.instances.UpdateShieldedInstanceConfig(request)
    return holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')
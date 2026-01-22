from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def MakeStatefulPolicyPreservedStateDiskEntry(messages, stateful_disk_dict):
    """Create StatefulPolicyPreservedState from a list of device names."""
    disk_device = messages.StatefulPolicyPreservedStateDiskDevice()
    if stateful_disk_dict.get('auto-delete'):
        disk_device.autoDelete = stateful_disk_dict.get('auto-delete').GetAutoDeleteEnumValue(messages.StatefulPolicyPreservedStateDiskDevice.AutoDeleteValueValuesEnum)
    return messages.StatefulPolicyPreservedState.DisksValue.AdditionalProperty(key=stateful_disk_dict.get('device-name'), value=disk_device)
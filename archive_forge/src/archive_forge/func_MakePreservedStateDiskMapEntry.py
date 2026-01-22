from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def MakePreservedStateDiskMapEntry(messages, device_name, source, mode, auto_delete='never'):
    """Make a map entry for disks field in preservedState message."""
    mode_map = {'READ_ONLY': messages.PreservedStatePreservedDisk.ModeValueValuesEnum.READ_ONLY, 'READ_WRITE': messages.PreservedStatePreservedDisk.ModeValueValuesEnum.READ_WRITE}
    mode_map['ro'] = mode_map['READ_ONLY']
    mode_map['rw'] = mode_map['READ_WRITE']
    auto_delete_map = {'never': messages.PreservedStatePreservedDisk.AutoDeleteValueValuesEnum.NEVER, 'on-permanent-instance-deletion': messages.PreservedStatePreservedDisk.AutoDeleteValueValuesEnum.ON_PERMANENT_INSTANCE_DELETION}
    preserved_disk = messages.PreservedStatePreservedDisk(autoDelete=auto_delete_map[auto_delete], source=source)
    if mode:
        preserved_disk.mode = mode_map[mode]
    return messages.PreservedState.DisksValue.AdditionalProperty(key=device_name, value=preserved_disk)
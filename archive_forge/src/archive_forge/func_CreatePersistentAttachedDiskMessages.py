from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def CreatePersistentAttachedDiskMessages(resources, compute_client, csek_keys, disks, project, location, scope, container_mount_disk=None, use_disk_type_uri=True):
    """Returns a list of AttachedDisk messages and the boot disk's reference."""
    disks_messages = []
    messages = compute_client.messages
    compute = compute_client.apitools_client
    for disk in disks:
        name = disk.get('name')
        mode_value = disk.get('mode', 'rw')
        if mode_value == 'rw':
            mode = messages.AttachedDisk.ModeValueValuesEnum.READ_WRITE
        else:
            mode = messages.AttachedDisk.ModeValueValuesEnum.READ_ONLY
        boot = disk.get('boot', False)
        auto_delete = disk.get('auto-delete', False)
        if 'scope' in disk and disk['scope'] == 'regional':
            scope = compute_scopes.ScopeEnum.REGION
        else:
            scope = compute_scopes.ScopeEnum.ZONE
        disk_ref = instance_utils.ParseDiskResource(resources, name, project, location, scope)
        force_attach = disk.get('force-attach')
        if csek_keys:
            disk_key_or_none = csek_utils.MaybeLookupKeyMessage(csek_keys, disk_ref, compute)
            kwargs = {'diskEncryptionKey': disk_key_or_none}
        else:
            kwargs = {}
        device_name = instance_utils.GetDiskDeviceName(disk, name, container_mount_disk)
        source = disk_ref.SelfLink()
        if scope == compute_scopes.ScopeEnum.ZONE and (not use_disk_type_uri):
            source = name
        attached_disk = messages.AttachedDisk(autoDelete=auto_delete, boot=boot, deviceName=device_name, mode=mode, source=source, type=messages.AttachedDisk.TypeValueValuesEnum.PERSISTENT, forceAttach=force_attach, **kwargs)
        if boot:
            disks_messages = [attached_disk] + disks_messages
        else:
            disks_messages.append(attached_disk)
    return disks_messages
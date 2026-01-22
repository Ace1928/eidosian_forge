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
def _CreateLocalSsdMessage(resources, messages, device_name, interface, size_bytes=None, location=None, scope=None, project=None, use_disk_type_uri=True):
    """Create a message representing a local ssd."""
    if location and use_disk_type_uri:
        disk_type_ref = instance_utils.ParseDiskType(resources, 'local-ssd', project, location, scope)
        disk_type = disk_type_ref.SelfLink()
    else:
        disk_type = 'local-ssd'
    maybe_interface_enum = messages.AttachedDisk.InterfaceValueValuesEnum(interface) if interface else None
    local_ssd = messages.AttachedDisk(type=messages.AttachedDisk.TypeValueValuesEnum.SCRATCH, autoDelete=True, deviceName=device_name, interface=maybe_interface_enum, mode=messages.AttachedDisk.ModeValueValuesEnum.READ_WRITE, initializeParams=messages.AttachedDiskInitializeParams(diskType=disk_type))
    if size_bytes is not None:
        local_ssd.diskSizeGb = utils.BytesToGb(size_bytes)
    return local_ssd
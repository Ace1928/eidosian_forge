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
def CreateDefaultBootAttachedDiskMessage(compute_client, resources, disk_type, disk_device_name, disk_auto_delete, disk_size_gb, require_csek_key_create, image_uri, instance_name, project, location, scope, csek_keys=None, kms_args=None, enable_kms=False, snapshot_uri=None, use_disk_type_uri=True, disk_provisioned_iops=None, disk_provisioned_throughput=None, instant_snapshot_uri=None, support_source_instant_snapshot=False):
    """Returns an AttachedDisk message for creating a new boot disk."""
    messages = compute_client.messages
    compute = compute_client.apitools_client
    if disk_type:
        if use_disk_type_uri:
            disk_type_ref = instance_utils.ParseDiskType(resources, disk_type, project, location, scope)
            disk_type = disk_type_ref.SelfLink()
    else:
        disk_type = None
    if csek_keys:
        effective_boot_disk_name = disk_device_name or instance_name
        disk_ref = resources.Parse(effective_boot_disk_name, collection='compute.disks', params={'project': project, 'zone': location})
        disk_key_or_none = csek_utils.MaybeToMessage(csek_keys.LookupKey(disk_ref, require_csek_key_create), compute)
        [image_key_or_none] = csek_utils.MaybeLookupKeyMessagesByUri(csek_keys, resources, [image_uri], compute)
        kwargs_init_parms = {'sourceImageEncryptionKey': image_key_or_none}
        kwargs_disk = {'diskEncryptionKey': disk_key_or_none}
    else:
        kwargs_disk = {}
        kwargs_init_parms = {}
        effective_boot_disk_name = disk_device_name
    if enable_kms:
        kms_key = kms_utils.MaybeGetKmsKey(kms_args, messages, kwargs_disk.get('diskEncryptionKey', None), boot_disk_prefix=True)
        if kms_key:
            kwargs_disk = {'diskEncryptionKey': kms_key}
    initialize_params = messages.AttachedDiskInitializeParams(sourceImage=image_uri, diskSizeGb=disk_size_gb, diskType=disk_type, **kwargs_init_parms)
    if disk_provisioned_iops is not None:
        initialize_params.provisionedIops = disk_provisioned_iops
    if disk_provisioned_throughput is not None:
        initialize_params.provisionedThroughput = disk_provisioned_throughput
    if snapshot_uri:
        initialize_params.sourceImage = None
        if support_source_instant_snapshot:
            initialize_params.sourceInstantSnapshot = None
        initialize_params.sourceSnapshot = snapshot_uri
    elif instant_snapshot_uri:
        initialize_params.sourceImage = None
        initialize_params.sourceSnapshot = None
        initialize_params.sourceInstantSnapshot = instant_snapshot_uri
    return messages.AttachedDisk(autoDelete=disk_auto_delete, boot=True, deviceName=effective_boot_disk_name, initializeParams=initialize_params, mode=messages.AttachedDisk.ModeValueValuesEnum.READ_WRITE, type=messages.AttachedDisk.TypeValueValuesEnum.PERSISTENT, **kwargs_disk)
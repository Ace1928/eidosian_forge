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
def CreateDiskMessages(args, project, location, scope, compute_client, resource_parser, image_uri, holder=None, boot_disk_size_gb=None, instance_name=None, create_boot_disk=False, csek_keys=None, support_kms=False, support_nvdimm=False, support_source_snapshot_csek=False, support_boot_snapshot_uri=False, support_image_csek=False, support_match_container_mount_disks=False, support_create_disk_snapshots=False, support_persistent_attached_disks=True, support_replica_zones=False, use_disk_type_uri=True, support_multi_writer=False, support_source_instant_snapshot=False, support_boot_instant_snapshot_uri=False, support_enable_confidential_compute=False):
    """Creates disk messages for a single instance."""
    container_mount_disk = []
    if support_match_container_mount_disks:
        container_mount_disk = args.container_mount_disk
    persistent_disks = []
    if support_persistent_attached_disks:
        persistent_disks = CreatePersistentAttachedDiskMessages(resources=resource_parser, compute_client=compute_client, csek_keys=csek_keys, disks=args.disk or [], project=project, location=location, scope=scope, container_mount_disk=container_mount_disk, use_disk_type_uri=use_disk_type_uri)
    persistent_create_disks = CreatePersistentCreateDiskMessages(compute_client=compute_client, resources=resource_parser, csek_keys=csek_keys, create_disks=getattr(args, 'create_disk', []), project=project, location=location, scope=scope, holder=holder, enable_kms=support_kms, enable_snapshots=support_create_disk_snapshots, container_mount_disk=container_mount_disk, enable_source_snapshot_csek=support_source_snapshot_csek, enable_image_csek=support_image_csek, support_replica_zones=support_replica_zones, use_disk_type_uri=use_disk_type_uri, support_multi_writer=support_multi_writer, enable_source_instant_snapshots=support_source_instant_snapshot, support_enable_confidential_compute=support_enable_confidential_compute)
    local_nvdimms = []
    if support_nvdimm:
        local_nvdimms = CreateLocalNvdimmMessages(args, resource_parser, compute_client.messages, location, scope, project)
    local_ssds = CreateLocalSsdMessages(args, resource_parser, compute_client.messages, location, scope, project, use_disk_type_uri)
    if create_boot_disk:
        boot_snapshot_uri = None
        if support_boot_snapshot_uri:
            boot_snapshot_uri = instance_utils.ResolveSnapshotURI(user_project=project, snapshot=args.source_snapshot, resource_parser=resource_parser)
        boot_instant_snapshot_uri = None
        if support_boot_instant_snapshot_uri:
            boot_instant_snapshot_uri = instance_utils.ResolveInstantSnapshotURI(user_project=project, instant_snapshot=args.source_instant_snapshot, resource_parser=resource_parser)
        boot_disk = CreateDefaultBootAttachedDiskMessage(compute_client=compute_client, resources=resource_parser, disk_type=args.boot_disk_type, disk_device_name=args.boot_disk_device_name, disk_auto_delete=args.boot_disk_auto_delete, disk_size_gb=boot_disk_size_gb, require_csek_key_create=args.require_csek_key_create if csek_keys else None, image_uri=image_uri, instance_name=instance_name, project=project, location=location, scope=scope, enable_kms=support_kms, csek_keys=csek_keys, kms_args=args, snapshot_uri=boot_snapshot_uri, disk_provisioned_iops=args.boot_disk_provisioned_iops, disk_provisioned_throughput=args.boot_disk_provisioned_throughput, use_disk_type_uri=use_disk_type_uri, instant_snapshot_uri=boot_instant_snapshot_uri, support_source_instant_snapshot=support_source_instant_snapshot)
        persistent_disks = [boot_disk] + persistent_disks
    if persistent_create_disks and persistent_create_disks[0].boot:
        boot_disk = persistent_create_disks.pop(0)
        persistent_disks = [boot_disk] + persistent_disks
    return persistent_disks + persistent_create_disks + local_nvdimms + local_ssds
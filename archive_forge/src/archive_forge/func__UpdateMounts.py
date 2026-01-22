from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _UpdateMounts(holder, manifest, remove_container_mounts, container_mount_host_path, container_mount_tmpfs, container_mount_disk, disks):
    """Updates mounts in container manifest."""
    _CleanupMounts(manifest, remove_container_mounts, container_mount_host_path, container_mount_tmpfs, container_mount_disk=container_mount_disk)
    used_names = [volume['name'] for volume in manifest['spec']['volumes']]
    volumes = []
    volume_mounts = []
    next_volume_index = 0
    for volume in container_mount_host_path:
        while _GetHostPathDiskName(next_volume_index) in used_names:
            next_volume_index += 1
        name = _GetHostPathDiskName(next_volume_index)
        next_volume_index += 1
        volumes.append({'name': name, 'hostPath': {'path': volume['host-path']}})
        volume_mounts.append({'name': name, 'mountPath': volume['mount-path'], 'readOnly': volume.get('mode', _DEFAULT_MODE).isReadOnly()})
    for tmpfs in container_mount_tmpfs:
        while _GetTmpfsDiskName(next_volume_index) in used_names:
            next_volume_index += 1
        name = _GetTmpfsDiskName(next_volume_index)
        next_volume_index += 1
        volumes.append({'name': name, 'emptyDir': {'medium': 'Memory'}})
        volume_mounts.append({'name': name, 'mountPath': tmpfs['mount-path']})
    if container_mount_disk:
        disks = [{'device-name': disk.deviceName, 'name': holder.resources.Parse(disk.source).Name()} for disk in disks]
        _AddMountedDisksToManifest(container_mount_disk, volumes, volume_mounts, used_names=used_names, disks=disks)
    manifest['spec']['containers'][0]['volumeMounts'].extend(volume_mounts)
    manifest['spec']['volumes'].extend(volumes)
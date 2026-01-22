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
def _CleanupMounts(manifest, remove_container_mounts, container_mount_host_path, container_mount_tmpfs, container_mount_disk=None):
    """Remove all specified mounts from container manifest."""
    container_mount_disk = container_mount_disk or []
    mount_paths_to_remove = remove_container_mounts[:]
    for host_path in container_mount_host_path:
        mount_paths_to_remove.append(host_path['mount-path'])
    for tmpfs in container_mount_tmpfs:
        mount_paths_to_remove.append(tmpfs['mount-path'])
    for disk in container_mount_disk:
        mount_paths_to_remove.append(disk['mount-path'])
    used_mounts = []
    used_mounts_names = []
    removed_mount_names = []
    for mount in manifest['spec']['containers'][0].get('volumeMounts', []):
        if mount['mountPath'] not in mount_paths_to_remove:
            used_mounts.append(mount)
            used_mounts_names.append(mount['name'])
        else:
            removed_mount_names.append(mount['name'])
    manifest['spec']['containers'][0]['volumeMounts'] = used_mounts
    used_volumes = []
    for volume in manifest['spec'].get('volumes', []):
        if volume['name'] in used_mounts_names or volume['name'] not in removed_mount_names:
            used_volumes.append(volume)
    manifest['spec']['volumes'] = used_volumes
from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import argparse
import collections
from collections.abc import Collection, Container, Iterable, Mapping, MutableMapping
import copy
import dataclasses
import itertools
import json
import types
from typing import Any, ClassVar
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
import six
@dataclasses.dataclass(frozen=True)
class SecretVolumeChanges(TemplateConfigChanger):
    """Represents the user intent to change volumes with secret source types.

  Attributes:
    updates: Updates to mount path and volume fields.
    removes: List of mount paths to remove.
    clear_others: If true clear all non-updated volumes and mounts of the given
      [volume_type].
    container_name: Name of the container to update.
  """
    updates: Mapping[str, secrets_mapping.ReachableSecret]
    removes: Collection[str]
    clear_others: bool
    container_name: str | None = None

    def _UpdateManagedVolumes(self, resource, volume_mounts, res_volumes, external_mounts):
        """Update volumes for Cloud Run. Ensure only one secret per directory."""
        new_volumes = {}
        volumes_to_mounts = collections.defaultdict(list)
        for path, vol in volume_mounts.items():
            volumes_to_mounts[vol].append(path)
        for file_path, reachable_secret in self.updates.items():
            mount_point = file_path.rsplit('/', 1)[0]
            if mount_point in new_volumes:
                if new_volumes[mount_point].secretName != reachable_secret.secret_name:
                    raise exceptions.ConfigurationError('Cannot update secret at [{}] because a different secret is already mounted in the same directory.'.format(file_path))
                reachable_secret.AppendToSecretVolumeSource(resource, new_volumes[mount_point])
            else:
                new_volumes[mount_point] = reachable_secret.AsSecretVolumeSource(resource)
        for mount_point, volume_source in new_volumes.items():
            if mount_point in volume_mounts:
                volume_name = volume_mounts[mount_point]
                if len(volumes_to_mounts[volume_name]) > 1 or volume_name in external_mounts:
                    volumes_to_mounts[volume_name].remove(mount_point)
                    new_name = _CopyToNewVolume(resource, volume_name, mount_point, volume_source, res_volumes, volume_mounts)
                    volumes_to_mounts[new_name].append(mount_point)
                    continue
                else:
                    volume = res_volumes[volume_name]
                    if volume.secretName != volume_source.secretName:
                        existing_paths = {item.path for item in volume.items}
                        new_paths = {item.path for item in volume_source.items}
                        if not existing_paths.issubset(new_paths):
                            raise exceptions.ConfigurationError('Multiple secrets are specified for directory [{}]. Cloud Run only supports one secret per directory'.format(mount_point))
                    else:
                        new_paths = {item.path for item in volume_source.items}
                        for item in volume.items:
                            if item.path not in new_paths:
                                volume_source.items.append(item)
            else:
                volume_name = _UniqueVolumeName(volume_source.secretName, resource.template.volumes)
                try:
                    volume_mounts[mount_point] = volume_name
                except KeyError:
                    raise exceptions.ConfigurationError('Cannot update mount [{}] because its mounted volume is of a different source type.'.format(mount_point))
            res_volumes[volume_name] = volume_source

    def Adjust(self, resource):
        """Mutates the given config's volumes to match the desired changes.

    Args:
      resource: k8s_object to adjust

    Returns:
      The adjusted resource

    Raises:
      ConfigurationError if there's an attempt to replace the volume a mount
        points to whose existing volume has a source of a different type than
        the new volume (e.g. mount that points to a volume with a secret source
        can't be replaced with a volume that has a config map source).
    """
        if self.container_name:
            container = resource.template.containers[self.container_name]
        else:
            container = resource.template.container
        volume_mounts = container.volume_mounts.secrets
        res_volumes = resource.template.volumes.secrets
        external_mounts = frozenset(itertools.chain.from_iterable((external_container.volume_mounts.secrets.values() for name, external_container in resource.template.containers.items() if name != container.name)))
        if platforms.IsManaged():
            _PruneManagedVolumeMapping(resource, res_volumes, volume_mounts, self.removes, self.clear_others, external_mounts)
        else:
            removes = self.removes
            _PruneMapping(volume_mounts, removes, self.clear_others)
        if platforms.IsManaged():
            self._UpdateManagedVolumes(resource, volume_mounts, res_volumes, external_mounts)
        else:
            for file_path, reachable_secret in self.updates.items():
                volume_name = _UniqueVolumeName(reachable_secret.secret_name, resource.template.volumes)
                try:
                    mount_point = file_path
                    volume_mounts[mount_point] = volume_name
                except KeyError:
                    raise exceptions.ConfigurationError('Cannot update mount [{}] because its mounted volume is of a different source type.'.format(file_path))
                res_volumes[volume_name] = reachable_secret.AsSecretVolumeSource(resource)
        _PruneVolumes(external_mounts.union(volume_mounts.values()), res_volumes)
        secrets_mapping.PruneAnnotation(resource)
        return resource
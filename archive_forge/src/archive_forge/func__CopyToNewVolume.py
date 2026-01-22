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
def _CopyToNewVolume(resource, volume_name, mount_point, volume_source, res_volumes, volume_mounts):
    """Copies existing volume to volume with a new name."""
    new_volume_name = _UniqueVolumeName(volume_source.secretName, resource.template.volumes)
    try:
        volume_mounts[mount_point] = new_volume_name
    except KeyError:
        raise exceptions.ConfigurationError('Cannot update mount [{}] because its mounted volume is of a different source type.'.format(mount_point))
    new_paths = {item.path for item in volume_source.items}
    old_volume = res_volumes[volume_name]
    for item in old_volume.items:
        if item.path not in new_paths:
            volume_source.items.append(item)
    res_volumes[new_volume_name] = volume_source
    return new_volume_name
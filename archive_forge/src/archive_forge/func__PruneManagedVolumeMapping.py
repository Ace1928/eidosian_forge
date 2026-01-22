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
def _PruneManagedVolumeMapping(resource, res_volumes, volume_mounts: MutableMapping[str, str], removes: Collection[str], clear_others: bool, external_mounts: Container[str]):
    """Remove the specified volume mappings from the config."""
    if clear_others:
        volume_mounts.clear()
    else:
        for remove in removes:
            mount, path = remove.rsplit('/', 1)
            if mount in volume_mounts:
                volume_name = volume_mounts[mount]
                if volume_name in external_mounts:
                    volume_name = _CopyToNewVolume(resource, volume_name, mount, copy.deepcopy(res_volumes[volume_name]), res_volumes, volume_mounts)
                new_paths = []
                for key_to_path in res_volumes[volume_name].items:
                    if path != key_to_path.path:
                        new_paths.append(key_to_path)
                if not new_paths:
                    del volume_mounts[mount]
                else:
                    res_volumes[volume_name].items = new_paths
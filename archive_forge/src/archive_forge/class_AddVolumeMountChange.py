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
class AddVolumeMountChange(ContainerConfigChanger):
    """Updates Volume Mounts set on the container.

  Attributes:
    new_mounts: Mounts to add to the adjusted container.
  """
    new_mounts: Collection[Mapping[str, str]] = dataclasses.field(default_factory=list)

    def AdjustContainer(self, container, messages_mod):
        for mount in self.new_mounts:
            if 'volume' not in mount or 'mount-path' not in mount:
                raise exceptions.ConfigurationError('Added Volume mounts must have a `volume` and a `mount-path`.')
            container.volume_mounts[mount['mount-path']] = mount['volume']
        return container
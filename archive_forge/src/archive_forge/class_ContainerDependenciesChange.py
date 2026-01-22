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
class ContainerDependenciesChange(TemplateConfigChanger):
    """Sets container dependencies.

  Updates container dependencies to add the dependencies in new_depencies.
  Additionally, dependencies to or from a container which does not exist will be
  removed.

  Attributes:
      new_dependencies: A map of containers to their updated dependencies.
        Defaults to an empty map.
  """
    new_dependencies: Mapping[str, Iterable[str]] = dataclasses.field(default_factory=dict)

    def Adjust(self, resource: k8s_object.KubernetesObject) -> k8s_object.KubernetesObject:
        containers = frozenset(resource.template.containers.keys())
        dependencies = resource.template.dependencies
        dependencies = {container_name: [c for c in depends_on if c in containers] for container_name, depends_on in dependencies.items() if container_name in containers}
        for container, depends_on in self.new_dependencies.items():
            if not container:
                container = resource.template.container.name
            depends_on = frozenset(depends_on)
            missing = depends_on - containers
            if missing:
                raise exceptions.ConfigurationError('--depends_on for container {} references nonexistent containers: {}.'.format(container, ','.join(missing)))
            if depends_on:
                dependencies[container] = sorted(depends_on)
            else:
                del dependencies[container]
        resource.template.dependencies = dependencies
        return resource
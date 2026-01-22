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
class ContainerConfigChanger(TemplateConfigChanger):
    """An abstract class representing container configuration changes.

  Attributes:
    container_name: Name of the container to modify. If None the primary
      container is modified.
  """
    container_name: str | None = None

    @abc.abstractmethod
    def AdjustContainer(self, container: container_resource.Container, messages_mod: types.ModuleType):
        """Mutate the given container.

    This method is called by this class's Adjust method and should apply the
    desired changes directly to container.

    Args:
      container: the container to adjust.
      messages_mod: Run v1 messages module.
    """

    def Adjust(self, resource: container_resource.ContainerResource):
        """Returns a modified resource.

    Adjusts resource by applying changes to the container specified by
    self.container_name if present or the primary container otherwise. Calls
    AdjustContainer to apply changes to the selected container.

    Args:
      resource: The resoure to modify.
    """
        if self.container_name:
            container = resource.template.containers[self.container_name]
        else:
            container = resource.template.container
        self.AdjustContainer(container, resource.MessagesModule())
        return resource
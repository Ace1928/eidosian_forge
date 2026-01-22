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
class RevisionNameChanges(TemplateConfigChanger):
    """Represents the user intent to change revision name.

  Attributes:
    revision_suffix: Suffix to append to the revision name.
  """
    revision_suffix: str

    def Adjust(self, resource):
        """Mutates the given config's revision name to match what's desired."""
        max_prefix_length = _MAX_RESOURCE_NAME_LENGTH - len(self.revision_suffix) - 1
        resource.template.name = '{}-{}'.format(resource.name[:max_prefix_length], self.revision_suffix)
        return resource
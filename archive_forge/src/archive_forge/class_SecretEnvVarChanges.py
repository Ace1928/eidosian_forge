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
class SecretEnvVarChanges(TemplateConfigChanger):
    """Represents the user intent to modify environment variable secrets.

  Attributes:
    updates: Env var names and values to update.
    removes: List of env vars to remove.
    clear_others: If true clear all non-updated env vars.
    container_name: Name of the container to update. If None, the resource's
      primary container is update.
  """
    updates: Mapping[str, secrets_mapping.ReachableSecret]
    removes: Collection[str]
    clear_others: bool
    container_name: str | None = None

    def Adjust(self, resource):
        """Mutates the given config's env vars to match the desired changes.

    Args:
      resource: k8s_object to adjust

    Returns:
      The adjusted resource

    Raises:
      ConfigurationError if there's an attempt to replace the source of an
        existing environment variable whose source is of a different type
        (e.g. env var's secret source can't be replaced with a config map
        source).
    """
        if self.container_name:
            env_vars = resource.template.containers[self.container_name].env_vars.secrets
        else:
            env_vars = resource.template.env_vars.secrets
        _PruneMapping(env_vars, self.removes, self.clear_others)
        for name, reachable_secret in self.updates.items():
            try:
                env_vars[name] = reachable_secret.AsEnvVarSource(resource)
            except KeyError:
                raise exceptions.ConfigurationError('Cannot update environment variable [{}] to the given type because it has already been set with a different type.'.format(name))
        secrets_mapping.PruneAnnotation(resource)
        return resource
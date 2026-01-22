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
class SetClientNameAndVersionAnnotationChange(ConfigChanger):
    """Sets the client name and version annotations.

  Attributes:
    client_name: Client name to set.
    client_version: Client version to set.
    set_on_template: A boolean indicating whether the client name and version
      annotations should be set on the resource template as well.
  """
    client_name: str
    client_version: str
    set_on_template: bool = True

    @property
    def adjusts_template(self):
        return self.set_on_template

    def Adjust(self, resource):
        if self.client_name is not None:
            resource.annotations[k8s_object.CLIENT_NAME_ANNOTATION] = self.client_name
            if self.set_on_template and hasattr(resource.template, 'annotations'):
                resource.template.annotations[k8s_object.CLIENT_NAME_ANNOTATION] = self.client_name
        if self.client_version is not None:
            resource.annotations[k8s_object.CLIENT_VERSION_ANNOTATION] = self.client_version
            if self.set_on_template and hasattr(resource.template, 'annotations'):
                resource.template.annotations[k8s_object.CLIENT_VERSION_ANNOTATION] = self.client_version
        return resource
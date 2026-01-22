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
class NetworkInterfacesChange(TemplateConfigChanger):
    """Sets or updates the network interfaces annotation on the template.

  Attributes:
    network_is_set: Boolean indicating whether network was explicitly set by the
      user.
    network: The network to set.
    subnet_is_set: Boolean indicating whether subnet was explicitly set by the
      user.
    subnet: The subnet to set.
    network_tags_is_set: Boolean indicating whether network_tags was explicitly
      set by the user.
    network_tags: The network tags to set.
  """
    network_is_set: bool
    network: str
    subnet_is_set: bool
    subnet: str
    network_tags_is_set: bool
    network_tags: list[str]

    def _SetOrClear(self, m, key, value):
        if value:
            m[key] = value
        elif key in m:
            del m[key]

    def Adjust(self, resource):
        annotations = resource.template.annotations
        network_interface = {}
        if k8s_object.NETWORK_INTERFACES_ANNOTATION in annotations:
            network_interface = json.loads(annotations[k8s_object.NETWORK_INTERFACES_ANNOTATION])[0]
        if self.network_is_set:
            self._SetOrClear(network_interface, 'network', self.network)
        if self.subnet_is_set:
            self._SetOrClear(network_interface, 'subnetwork', self.subnet)
        if self.network_tags_is_set:
            self._SetOrClear(network_interface, 'tags', self.network_tags)
        value = ''
        if network_interface:
            value = '[{interfaces}]'.format(interfaces=json.dumps(network_interface, sort_keys=True))
        self._SetOrClear(annotations, k8s_object.NETWORK_INTERFACES_ANNOTATION, value)
        if not value and container_resource.EGRESS_SETTINGS_ANNOTATION in annotations:
            del annotations[container_resource.EGRESS_SETTINGS_ANNOTATION]
        return resource
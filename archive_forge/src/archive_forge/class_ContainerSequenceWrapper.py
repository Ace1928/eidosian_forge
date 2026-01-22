from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
class ContainerSequenceWrapper(collections_abc.MutableSequence):
    """Wraps a list of containers wrapping each element with the Container wrapper class."""

    def __init__(self, containers_to_wrap, volumes, messages_mod):
        super(ContainerSequenceWrapper, self).__init__()
        self._containers = containers_to_wrap
        self._volumes = volumes
        self._messages = messages_mod

    def __getitem__(self, index):
        return Container(self._volumes, self._messages, self._containers[index])

    def __len__(self):
        return len(self._containers)

    def __setitem__(self, index, container):
        self._containers[index] = container.MakeSerializable()

    def __delitem__(self, index):
        del self._containers[index]

    def insert(self, index, value):
        self._containers.insert(index, value.MakeSerializable())

    def MakeSerializable(self):
        return self._containers
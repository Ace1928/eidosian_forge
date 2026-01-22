import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __GetDescriptorByName(self, name):
    if name in self.__message_registry:
        return self.__message_registry[name]
    if name in self.__nascent_types:
        raise ValueError('Cannot retrieve type currently being created: %s' % name)
    return None
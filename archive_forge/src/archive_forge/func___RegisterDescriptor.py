import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __RegisterDescriptor(self, new_descriptor):
    """Register the given descriptor in this registry."""
    if not isinstance(new_descriptor, (extended_descriptor.ExtendedMessageDescriptor, extended_descriptor.ExtendedEnumDescriptor)):
        raise ValueError('Cannot add descriptor of type %s' % (type(new_descriptor),))
    full_name = self.__ComputeFullName(new_descriptor.name)
    if full_name in self.__message_registry:
        raise ValueError('Attempt to re-register descriptor %s' % full_name)
    if full_name not in self.__nascent_types:
        raise ValueError('Directly adding types is not supported')
    new_descriptor.full_name = full_name
    self.__message_registry[full_name] = new_descriptor
    if isinstance(new_descriptor, extended_descriptor.ExtendedMessageDescriptor):
        self.__current_env.message_types.append(new_descriptor)
    elif isinstance(new_descriptor, extended_descriptor.ExtendedEnumDescriptor):
        self.__current_env.enum_types.append(new_descriptor)
    self.__unknown_types.discard(full_name)
    self.__nascent_types.remove(full_name)
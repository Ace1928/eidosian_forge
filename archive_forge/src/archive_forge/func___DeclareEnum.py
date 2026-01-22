import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __DeclareEnum(self, enum_name, attrs):
    description = util.CleanDescription(attrs.get('description', ''))
    enum_values = attrs['enum']
    enum_descriptions = attrs.get('enumDescriptions', [''] * len(enum_values))
    self.AddEnumDescriptor(enum_name, description, enum_values, enum_descriptions)
    self.__AddIfUnknown(enum_name)
    return TypeInfo(type_name=enum_name, variant=messages.Variant.ENUM)
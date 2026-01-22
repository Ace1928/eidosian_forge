import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def AddEnumDescriptor(self, name, description, enum_values, enum_descriptions):
    """Add a new EnumDescriptor named name with the given enum values."""
    message = extended_descriptor.ExtendedEnumDescriptor()
    message.name = self.__names.ClassName(name)
    message.description = util.CleanDescription(description)
    self.__DeclareDescriptor(message.name)
    for index, (enum_name, enum_description) in enumerate(zip(enum_values, enum_descriptions)):
        enum_value = extended_descriptor.ExtendedEnumValueDescriptor()
        enum_value.name = self.__names.NormalizeEnumName(enum_name)
        if enum_value.name != enum_name:
            message.enum_mappings.append(extended_descriptor.ExtendedEnumDescriptor.JsonEnumMapping(python_name=enum_value.name, json_name=enum_name))
            self.__AddImport('from %s import encoding' % self.__base_files_package)
        enum_value.number = index
        enum_value.description = util.CleanDescription(enum_description or '<no description>')
        message.values.append(enum_value)
    self.__RegisterDescriptor(message)
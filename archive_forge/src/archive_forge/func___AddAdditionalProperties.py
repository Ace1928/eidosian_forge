import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __AddAdditionalProperties(self, message, schema, properties):
    """Add an additionalProperties field to message."""
    additional_properties_info = schema['additionalProperties']
    entries_type_name = self.__AddAdditionalPropertyType(message.name, additional_properties_info)
    description = util.CleanDescription(additional_properties_info.get('description'))
    if description is None:
        description = 'Additional properties of type %s' % message.name
    attrs = {'items': {'$ref': entries_type_name}, 'description': description, 'type': 'array'}
    field_name = 'additionalProperties'
    message.fields.append(self.__FieldDescriptorFromProperties(field_name, len(properties) + 1, attrs))
    self.__AddImport('from %s import encoding' % self.__base_files_package)
    message.decorators.append('encoding.MapUnrecognizedFields(%r)' % field_name)
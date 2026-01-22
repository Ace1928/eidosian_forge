from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
def _ParseFieldType(self, field_type):
    """Parses a field type."""
    primitive_field_type_enum = self.messages.GoogleCloudDatacatalogV1FieldType.PrimitiveTypeValueValuesEnum
    valid_primitive_type_mapping = {'double': primitive_field_type_enum.DOUBLE, 'string': primitive_field_type_enum.STRING, 'bool': primitive_field_type_enum.BOOL, 'timestamp': primitive_field_type_enum.TIMESTAMP}
    if field_type in valid_primitive_type_mapping:
        return self.messages.GoogleCloudDatacatalogV1FieldType(primitiveType=valid_primitive_type_mapping[field_type])
    else:
        enum_values = self._ParseEnumValues(field_type)
        if enum_values:
            return self.messages.GoogleCloudDatacatalogV1FieldType(enumType=self.messages.GoogleCloudDatacatalogV1FieldTypeEnumType(allowedValues=enum_values))
    raise exceptions.InvalidArgumentException('--field', field_type)
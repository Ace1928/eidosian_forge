from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
def _ParseEnumValues(self, raw_enum_values):
    """Parses a raw enum value (e.g. 'enum(A|B|C)).

    Args:
      raw_enum_values: User-supplied definition of an enum

    Returns:
      DataCatalog FieldTypeEnumTypeEnumValue or none if a valid enum type wasn't
      not provided.
    """
    match = re.search('enum\\((.*)\\)', raw_enum_values, re.IGNORECASE)
    if not match:
        return None
    enum_values = []
    for enum in match.group(1).split('|'):
        enum_values.append(self._MakeEnumValue(enum))
    return enum_values
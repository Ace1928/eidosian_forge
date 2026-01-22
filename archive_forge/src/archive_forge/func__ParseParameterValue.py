from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def _ParseParameterValue(type_dict, value_input):
    """Parse a parameter value of type `type_dict` from value_input.

  Arguments:
    type_dict: The JSON-dict type as which to parse `value_input`.
    value_input: Either a string representing the value, or a JSON dict for
      array and value types.

  Returns:
    A dict with one of value, arrayValues, or structValues populated depending
    on the type.

  """
    if 'structTypes' in type_dict:
        if _IsString(value_input):
            if value_input == 'NULL':
                return {'structValues': None}
            value_input = json.loads(value_input)
        value_input = collections.OrderedDict(sorted(value_input.items()))
        type_map = collections.OrderedDict([(x['name'], x['type']) for x in type_dict['structTypes']])
        values = collections.OrderedDict()
        for field_name, value in six.iteritems(value_input):
            values[field_name] = _ParseParameterValue(type_map[field_name], value)
        return {'structValues': values}
    if 'arrayType' in type_dict:
        if _IsString(value_input):
            if value_input == 'NULL':
                return {'arrayValues': None}
            value_input = json.loads(value_input)
        values = [_ParseParameterValue(type_dict['arrayType'], x) for x in value_input]
        return {'arrayValues': values}
    return {'value': value_input if value_input != 'NULL' else None}
import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def _CamelCaseToSnakeCase(path_name):
    """Converts a field name from camelCase to snake_case."""
    result = []
    for c in path_name:
        if c == '_':
            raise ValueError('Fail to parse FieldMask: Path name {0} must not contain "_"s.'.format(path_name))
        if c.isupper():
            result += '_'
            result += c.lower()
        else:
            result += c
    return ''.join(result)
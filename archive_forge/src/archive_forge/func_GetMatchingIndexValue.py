from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def GetMatchingIndexValue(index, func):
    """Returns the first non-None func value for case-converted index."""
    value = func(index)
    if value:
        return value
    if not isinstance(index, six.string_types):
        return None
    for convert in [ConvertToCamelCase, ConvertToSnakeCase]:
        value = func(convert(index))
        if value:
            return value
    return None
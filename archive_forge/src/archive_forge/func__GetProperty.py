from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
from googlecloudsdk.core.util import tokenizer
import six
def _GetProperty(obj, components):
    """Grabs a property from obj."""
    if obj is None:
        return None
    elif not components:
        return obj
    elif isinstance(components[0], _Key) and isinstance(obj, dict):
        return _GetProperty(obj.get(components[0]), components[1:])
    elif isinstance(components[0], _Index) and isinstance(obj, list) and (components[0] < len(obj)):
        return _GetProperty(obj[components[0]], components[1:])
    elif isinstance(components[0], _Slice) and isinstance(obj, list):
        return [_GetProperty(item, components[1:]) for item in obj]
    else:
        return None
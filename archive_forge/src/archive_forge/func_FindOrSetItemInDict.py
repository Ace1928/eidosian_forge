from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def FindOrSetItemInDict(item, item_path, item_sep='.', set_value=None):
    """Finds (potentially) nested value based on specified node_path.

  If set_value is passed will set the value at item_path,
  creating if needed.
  Args:
      item: Dict, Map like object to search.
      item_path: str, An item_sep separated path to nested item in map.
      item_sep: str, Path item separator, default is '.'.
      set_value: object, value to set at item_path. If path is not found
        create a new item at item_path with value of set_value.

  Returns:
      Object, data found in input item at item_path or None.

  Raises:
    KeyError: If item_path not found or empty.
  """
    if not item_path:
        raise KeyError(item_path)
    parts = item_path.split(item_sep)
    parts.reverse()
    context = item
    while parts:
        part = parts.pop()
        if part in context and yaml.dict_like(context):
            if set_value and (not parts):
                context[part] = set_value
            context = context.get(part)
        elif set_value and yaml.dict_like(context):
            if parts:
                context[part] = collections.OrderedDict()
                context = context.get(part)
            else:
                context[part] = set_value
        else:
            raise KeyError('Path [{}] not found'.format(item_path))
    return context
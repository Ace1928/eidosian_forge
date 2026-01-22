from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _FindOrSetItem(self, item_path, item_sep='.', set_value=None):
    """Finds (potentially) nested value based on specified item_path.

    Args:
        item_path: str, An item_sep separated path to nested item in map.
        item_sep: str, Path item separator, default is '.'.
        set_value: object, value to set at item_path. If path is not found
          create a new item at item_path with value of set_value.

    Returns:
        Object, item found in map at item_path or None.
    """
    return FindOrSetItemInDict(self._content, item_path, item_sep, set_value)
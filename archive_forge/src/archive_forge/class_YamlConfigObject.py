from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
class YamlConfigObject(collections_abc.MutableMapping):
    """Abstraction for managing resource configuration Object.

  Attributes:
    content: YAMLObject, The parsed underlying config data.
  """

    def __init__(self, content):
        self._content = content

    @property
    def content(self):
        return copy.deepcopy(self._content)

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

    def __str__(self):
        yaml.convert_to_block_text(self._content)
        return yaml.dump(self._content, round_trip=True)

    def __setitem__(self, key, value):
        self._FindOrSetItem(key, set_value=value)

    def __getitem__(self, key):
        return self._FindOrSetItem(key)

    def __delitem__(self, key):
        DeleteItemInDict(self._content, key)

    def __iter__(self):
        return iter(self._content)

    def __len__(self):
        return len(self._content)

    def __contains__(self, key_path):
        try:
            self._FindOrSetItem(key_path)
        except KeyError:
            return False
        return True
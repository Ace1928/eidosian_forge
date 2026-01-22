from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from collections import OrderedDict
import re
from apitools.base.py import extra_types
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import zip
def GetJsonKeys(self, keys_list):
    """Get the primary key values to be written from keys input.

    Args:
      keys_list: String list, the primary key values of the row to be deleted.

    Returns:
      List of extra_types.JsonValue.

    Raises:
      InvalidKeysError: the keys are invalid.
    """
    if len(keys_list) != len(self._primary_keys):
        raise InvalidKeysError('Invalid keys. There are {} primary key columns in the table [{}]. {} are given.'.format(len(self._primary_keys), self.name, len(keys_list)))
    keys_json_list = []
    for key_name, key_value in zip(self._primary_keys, keys_list):
        col_in_table = self._FindColumnByName(key_name)
        col_json_value = col_in_table.GetJsonValues(key_value)
        keys_json_list.append(col_json_value)
    return keys_json_list
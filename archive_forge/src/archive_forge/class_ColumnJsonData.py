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
class ColumnJsonData(object):
    """Container for the column name and value to be written in a table.

  Attributes:
    col_name: String, the name of the column to be written.
    col_value: extra_types.JsonArray(array column) or
      extra_types.JsonValue(scalar column), the value to be written.
  """

    def __init__(self, col_name, col_value):
        self.col_name = col_name
        self.col_value = col_value
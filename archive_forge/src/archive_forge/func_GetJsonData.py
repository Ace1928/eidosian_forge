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
def GetJsonData(self, data_dict):
    """Get the column names and values to be written from data input.

    Args:
      data_dict: Dictionary where keys are the column names and values are user
          input data value, which is parsed from --data argument in the command.

    Returns:
      List of ColumnJsonData, which includes the column names and values to be
        written.
    """
    column_list = []
    for col_name, col_value in six.iteritems(data_dict):
        col_in_table = self._FindColumnByName(col_name)
        col_json_value = col_in_table.GetJsonValues(col_value)
        column_list.append(ColumnJsonData(col_name, col_json_value))
    return column_list
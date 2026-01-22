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
def GetColumnTypes(self):
    """Maps the column name to the column type.

    Returns:
      OrderedDict of column names to types.
    """
    col_to_type = OrderedDict()
    for name, column in six.iteritems(self._columns):
        col_to_type[name] = column.col_type
    return col_to_type
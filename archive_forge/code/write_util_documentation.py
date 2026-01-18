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
Find the _TableColumn object with the given column name.

    Args:
      col_name: String, the name of the column.

    Returns:
      _TableColumn.

    Raises:
      BadColumnNameError: the column name is invalid.
    
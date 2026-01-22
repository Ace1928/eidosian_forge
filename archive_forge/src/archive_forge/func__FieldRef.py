from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import gc
import os
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import metadata_table
from googlecloudsdk.core.cache import persistent_cache_base
from googlecloudsdk.core.util import files
import six
from six.moves import range  # pylint: disable=redefined-builtin
import sqlite3
def _FieldRef(column):
    """Returns a field reference name.

  Args:
    column: The field column number counting from 0.

  Returns:
    A field reference name.
  """
    return 'f{column}'.format(column=column)
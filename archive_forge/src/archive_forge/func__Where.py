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
def _Where(row_template=None):
    """Returns a WHERE clause for the row template.

  Column string matching supports * and ? match ops.

  Args:
    row_template: A template row tuple. A column value None means match all
      values for this column. A None value for row means all rows.

  Returns:
    A WHERE clause for the row template or the empty string if there is no none.
  """
    terms = []
    if row_template:
        for index in range(len(row_template)):
            term = row_template[index]
            if term is None:
                continue
            if isinstance(term, six.string_types):
                pattern = term.replace('*', '%').replace('.', '_').replace('"', '""')
                terms.append('{field} LIKE "{pattern}"'.format(field=_FieldRef(index), pattern=pattern))
            else:
                terms.append('{field} = {term}'.format(field=_FieldRef(index), term=term))
    if not terms:
        return ''
    return ' WHERE ' + ' AND '.join(terms)
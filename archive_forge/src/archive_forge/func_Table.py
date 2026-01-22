from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from googlecloudsdk.core.cache import exceptions
import six
import six.moves.urllib.parse
@abc.abstractmethod
def Table(self, name, create=True, columns=1, keys=1, timeout=None):
    """Returns the Table object for existing table name.

    Args:
      name: The table name.
      create: If True creates the table if it does not exist.
      columns: The number of columns in the table. Must be >= 1.
      keys: The number of columns, starting from 0, that form the primary
        row key. Must be 1 <= keys <= columns. The primary key is used to
        differentiate rows in the AddRows and DeleteRows methods.
      timeout: The table timeout in seconds, 0 for no timeout.

    Raises:
      CacheTableNameInvalid: name is not a valid table name.
      CacheTableNotFound: If the table does not exist.

    Returns:
      A Table object for name.
    """
    pass
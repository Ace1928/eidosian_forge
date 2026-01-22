from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from googlecloudsdk.core.cache import exceptions
import six
import six.moves.urllib.parse
@abc.abstractmethod
def DeleteRows(self, row_templates=None):
    """Deletes each row in the table matching any of the row_templates.

    Args:
      row_templates: A list of row templates. See Select() below for a detailed
        description of templates. None deletes all rows and is allowed for
        expired tables.
    """
    pass
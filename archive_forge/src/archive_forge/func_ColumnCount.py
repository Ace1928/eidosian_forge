from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def ColumnCount(self):
    """Returns the number of columns in the projection.

    Returns:
      The number of columns in the projection, 0 if the entire resource is
        projected.
    """
    return len(self._columns)
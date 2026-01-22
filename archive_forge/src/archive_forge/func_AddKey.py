from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def AddKey(self, key, attribute):
    """Adds key and attribute to the projection.

    Args:
      key: The parsed key to add.
      attribute: Parsed _Attribute to add.
    """
    self._columns.append(self._Column(key, attribute))
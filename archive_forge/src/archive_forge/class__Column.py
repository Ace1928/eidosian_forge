from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
class _Column(object):
    """Column key and transform attribute for self._columns.

    Attributes:
      key: The column key.
      attribute: The column key _Attribute.
    """

    def __init__(self, key, attribute):
        self.key = key
        self.attribute = attribute
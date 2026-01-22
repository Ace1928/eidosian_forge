from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def AddAlias(self, name, key, attribute):
    """Adds name as an alias for key and attribute to the projection.

    Args:
      name: The short (no dots) alias name for key.
      key: The parsed key to add.
      attribute: The attribute for key.
    """
    self.aliases[name] = (key, attribute)
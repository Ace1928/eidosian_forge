from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def SetRoot(self, root):
    """Sets the projection root node.

    Args:
      root: The resource_projector_parser._Tree root node.
    """
    self._tree = root
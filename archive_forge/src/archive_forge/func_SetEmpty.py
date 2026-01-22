from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def SetEmpty(self, node):
    """Sets the projector resource_projector_parser._Tree empty node.

    The empty node is used by to apply [] empty slice projections.

    Args:
      node: The projector resource_projector_parser._Tree empty node.
    """
    self._empty = node
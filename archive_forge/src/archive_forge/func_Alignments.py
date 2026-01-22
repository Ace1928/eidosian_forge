from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def Alignments(self):
    """Returns the projection column justfication list.

    Returns:
      The ordered list of alignment functions, where each function is one of
        ljust [default], center, or rjust.
    """
    return [ALIGNMENTS[col.attribute.align] for col in self._columns]
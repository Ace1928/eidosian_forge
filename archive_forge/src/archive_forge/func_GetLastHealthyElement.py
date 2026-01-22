from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def GetLastHealthyElement(self):
    """Returns the last element of the trace that is not an error.

    This element will contain the final component indicated by the trace.

    Returns:
      The last element of the trace that is not an error.
    """
    for element in reversed(self.elements):
        if not element.HasError():
            return element
    return None
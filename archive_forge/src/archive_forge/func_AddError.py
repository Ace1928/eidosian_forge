from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def AddError(self, error, args):
    element = FireTraceElement(error=error, args=args)
    self.elements.append(element)
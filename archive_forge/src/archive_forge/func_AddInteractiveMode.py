from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def AddInteractiveMode(self):
    element = FireTraceElement(action=INTERACTIVE_MODE)
    self.elements.append(element)
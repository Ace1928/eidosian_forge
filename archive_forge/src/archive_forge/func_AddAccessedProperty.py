from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def AddAccessedProperty(self, component, target, args, filename, lineno):
    element = FireTraceElement(component=component, action=ACCESSED_PROPERTY, target=target, args=args, filename=filename, lineno=lineno)
    self.elements.append(element)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def AddCompletionScript(self, script):
    element = FireTraceElement(component=script, action=COMPLETION_SCRIPT)
    self.elements.append(element)
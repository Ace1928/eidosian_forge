from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def AddCalledComponent(self, component, target, args, filename, lineno, capacity, action=CALLED_CALLABLE):
    """Adds an element to the trace indicating that a component was called.

    Also applies to instantiating a class.

    Args:
      component: The result of calling the callable.
      target: The name of the callable.
      args: The args consumed in order to call this callable.
      filename: The file in which the callable is defined, or None if N/A.
      lineno: The line number on which the callable is defined, or None if N/A.
      capacity: (bool) Whether the callable could have accepted additional args.
      action: The value to include as the action in the FireTraceElement.
    """
    element = FireTraceElement(component=component, action=action, target=target, args=args, filename=filename, lineno=lineno, capacity=capacity)
    self.elements.append(element)
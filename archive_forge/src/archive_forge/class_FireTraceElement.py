from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
class FireTraceElement(object):
    """A FireTraceElement represents a single step taken by a Fire execution.

  Examples of a FireTraceElement are the instantiation of a class or the
  accessing of an object member.
  """

    def __init__(self, component=None, action=None, target=None, args=None, filename=None, lineno=None, error=None, capacity=None):
        """Instantiates a FireTraceElement.

    Args:
      component: The result of this element of the trace.
      action: The type of action (eg instantiating a class) taking place.
      target: (string) The name of the component being acted upon.
      args: The args consumed by the represented action.
      filename: The file in which the action is defined, or None if N/A.
      lineno: The line number on which the action is defined, or None if N/A.
      error: The error represented by the action, or None if N/A.
      capacity: (bool) Whether the action could have accepted additional args.
    """
        self.component = component
        self._action = action
        self._target = target
        self.args = args
        self._filename = filename
        self._lineno = lineno
        self._error = error
        self._separator = False
        self._capacity = capacity

    def HasError(self):
        return self._error is not None

    def HasCapacity(self):
        return self._capacity

    def HasSeparator(self):
        return self._separator

    def AddSeparator(self):
        self._separator = True

    def ErrorAsStr(self):
        return ' '.join((str(arg) for arg in self._error.args))

    def __str__(self):
        if self.HasError():
            return self.ErrorAsStr()
        else:
            string = self._action
            if self._target is not None:
                string += ' "{target}"'.format(target=self._target)
            if self._filename is not None:
                path = self._filename
                if self._lineno is not None:
                    path += ':{lineno}'.format(lineno=self._lineno)
                string += ' ({path})'.format(path=path)
            return string
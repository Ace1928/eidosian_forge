from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def _Pop(self):
    """Pop values from stack at end of nesting.

    Called to indicate the end of a nested scope.

    Returns:
      Previously pushed value at the top of the stack.
    """
    assert self._stack != [] and self._stack is not None
    token, value = self._stack.pop()
    if self._stack:
        self._top = self._stack[-1]
    else:
        self._top = None
    return value
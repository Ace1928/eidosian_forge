from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
class DefaultArgTypeWrapper(ArgTypeUsage):
    """Base class for processing arg_type output but maintaining usage help text.

  Attributes:
    arg_type: type function used to parse input string into correct type
      ie ArgObject(value_type=int, repeating=true), int, bool, etc
  """

    def __init__(self, arg_type):
        super(DefaultArgTypeWrapper, self).__init__()
        self.arg_type = arg_type

    @property
    def _is_usage_type(self):
        return isinstance(self.arg_type, ArgTypeUsage)

    @property
    def hidden(self):
        if self._is_usage_type:
            return self.arg_type.hidden
        else:
            return None

    def GetUsageMetavar(self, *args, **kwargs):
        """Forwards default usage metavar for arg_type."""
        if self._is_usage_type:
            return self.arg_type.GetUsageMetavar(*args, **kwargs)
        else:
            return None

    def GetUsageExample(self, *args, **kwargs):
        """Forwards default usage example for arg_type."""
        if self._is_usage_type:
            return self.arg_type.GetUsageExample(*args, **kwargs)
        else:
            return None

    def GetUsageHelpText(self, *args, **kwargs):
        """Forwards default help text for arg_type."""
        if self._is_usage_type:
            return self.arg_type.GetUsageHelpText(*args, **kwargs)
        else:
            return None
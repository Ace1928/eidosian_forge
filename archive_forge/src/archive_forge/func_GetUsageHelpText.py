from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def GetUsageHelpText(self, *args, **kwargs):
    """Forwards default help text for arg_type."""
    if self._is_usage_type:
        return self.arg_type.GetUsageHelpText(*args, **kwargs)
    else:
        return None
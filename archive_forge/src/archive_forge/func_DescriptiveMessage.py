from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import sys
def DescriptiveMessage(self):
    """Descriptive message about what's happened to the last user operation."""
    if self._ready_condition and self._ready_condition in self and self[self._ready_condition]['message']:
        return self[self._ready_condition]['message']
    return None
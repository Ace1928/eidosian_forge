from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import sys
def NonTerminalSubconditions(self):
    """Yields keys of the conditions which do not directly affect Ready."""
    for k in self:
        if k != self._ready_condition and self[k]['severity'] and (self[k]['severity'] != _SEVERITY_ERROR):
            yield k
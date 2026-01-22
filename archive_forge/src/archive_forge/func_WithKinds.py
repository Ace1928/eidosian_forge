from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def WithKinds(self, kinds):
    """Add metadata kinds to this filter."""
    self._kinds = list(kinds)
    return self
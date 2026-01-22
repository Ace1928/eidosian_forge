from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def WithResources(self, resources):
    """Add resources to this filter."""
    self._resources = list(resources)
    return self
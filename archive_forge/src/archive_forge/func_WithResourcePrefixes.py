from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def WithResourcePrefixes(self, resource_prefixes):
    """Add resource prefixes to this filter."""
    self._resource_prefixes = list(resource_prefixes)
    return self
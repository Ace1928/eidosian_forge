from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ConvertToNamespaceName(name):
    """Convert name to namespace format (e.g. 'foo_bar')."""
    name = StripFlagPrefix(name)
    return name.lower().replace('-', '_').replace(' ', '_')
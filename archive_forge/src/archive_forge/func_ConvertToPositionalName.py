from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ConvertToPositionalName(name):
    """Convert name to positional format (e.g. 'FOO_BAR')."""
    name = StripFlagPrefix(name)
    return name.upper().replace('-', '_').replace(' ', '_')
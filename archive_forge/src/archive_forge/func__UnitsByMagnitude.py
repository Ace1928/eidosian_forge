from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import six
def _UnitsByMagnitude(units, type_abbr):
    """Returns a list of the units in scales sorted by magnitude."""
    scale_items = sorted(six.iteritems(units), key=lambda value: (value[1], value[0]))
    return [key + type_abbr for key, _ in scale_items if key]
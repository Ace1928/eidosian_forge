from __future__ import absolute_import
import re
def SecondsToDuration(value):
    """Convert seconds expressed as integer to a Duration value."""
    return '%ss' % int(value)
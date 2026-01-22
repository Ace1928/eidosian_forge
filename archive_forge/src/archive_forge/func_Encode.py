from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def Encode(s):
    """Return bytes objects encoded for HTTP headers / payload."""
    if s is None:
        return s
    return s.encode('utf-8')
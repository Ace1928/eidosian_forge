from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _AndJoinFilters(*filters):
    return ' AND '.join(['({})'.format(f) for f in filters if f])
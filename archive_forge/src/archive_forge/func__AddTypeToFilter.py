from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _AddTypeToFilter(filter_expr, channel_type):
    type_filter = 'type="{}"'.format(channel_type)
    if not filter_expr:
        return type_filter
    return '{0} AND ({1})'.format(type_filter, filter_expr)
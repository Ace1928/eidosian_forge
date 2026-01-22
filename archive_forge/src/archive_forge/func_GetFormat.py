from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import display_taps
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import cache_update_ops
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_reference
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import peek_iterable
import six
def GetFormat(self):
    """Determines the display format.

    Returns:
      format: The display format string.
    """
    default_fmt = self._GetDefaultFormat()
    fmt = self._GetExplicitFormat()
    if not fmt:
        if self._GetFlag('uri'):
            return 'value(.)'
        self._default_format_used = True
        fmt = default_fmt
    elif default_fmt:
        fmt = default_fmt + ' ' + fmt
    if not fmt:
        return fmt
    sort_keys = self._GetSortKeys()
    if not sort_keys:
        return fmt
    orders = []
    for order, (key, reverse) in enumerate(sort_keys, start=1):
        attr = ':reverse' if reverse else ''
        orders.append('{name}:sort={order}{attr}'.format(name=resource_lex.GetKeyName(key), order=order, attr=attr))
    fmt += ':({orders})'.format(orders=','.join(orders))
    return fmt
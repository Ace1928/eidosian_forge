from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def YieldFromListHandlingUnreachable(*args, **kwargs):
    """Yields from paged List calls handling unreachable."""
    unreachable = set()

    def _GetFieldFn(message, attr):
        unreachable.update(message.unreachable)
        return getattr(message, attr)
    result = list_pager.YieldFromList(*args, get_field_func=_GetFieldFn, **kwargs)
    for item in result:
        yield item
    if unreachable:
        log.warning('The following locations were unreachable: %s', ', '.join(sorted(unreachable)))
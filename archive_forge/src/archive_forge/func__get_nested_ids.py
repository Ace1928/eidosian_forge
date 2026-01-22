import sys
import time
from heatclient._i18n import _
from heatclient.common import utils
import heatclient.exc as exc
from heatclient.v1 import events as events_mod
def _get_nested_ids(hc, stack_id):
    nested_ids = []
    try:
        resources = hc.resources.list(stack_id=stack_id)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % stack_id)
    for r in resources:
        nested_id = utils.resource_nested_identifier(r)
        if nested_id:
            nested_ids.append(nested_id)
    return nested_ids
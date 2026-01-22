import sys
import time
from heatclient._i18n import _
from heatclient.common import utils
import heatclient.exc as exc
from heatclient.v1 import events as events_mod
def _get_stack_events(hc, stack_id, event_args):
    event_args['stack_id'] = stack_id
    try:
        events = hc.events.list(**event_args)
    except exc.HTTPNotFound as ex:
        raise exc.CommandError(str(ex))
    else:
        stack_name = stack_id.split('/')[0]
        for e in events:
            e.stack_name = _get_stack_name_from_links(e) or stack_name
        return events
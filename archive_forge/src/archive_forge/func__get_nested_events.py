import sys
import time
from heatclient._i18n import _
from heatclient.common import utils
import heatclient.exc as exc
from heatclient.v1 import events as events_mod
def _get_nested_events(hc, nested_depth, stack_id, event_args):
    nested_ids = _get_nested_ids(hc, stack_id)
    nested_events = []
    for n_id in nested_ids:
        stack_events = _get_stack_events(hc, n_id, event_args)
        if stack_events:
            nested_events.extend(stack_events)
        if nested_depth > 1:
            next_depth = nested_depth - 1
            nested_events.extend(_get_nested_events(hc, next_depth, n_id, event_args))
    return nested_events
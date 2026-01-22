import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import event_utils
from heatclient.common import format_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
def _stack_action(stack, parsed_args, heat_client, action, action_name=None):
    if parsed_args.wait:
        events = event_utils.get_events(heat_client, stack_id=stack, event_args={'sort_dir': 'desc'}, limit=1)
        marker = events[0].id if events else None
    try:
        action(stack)
    except heat_exc.HTTPNotFound:
        msg = _('Stack not found: %s') % stack
        raise exc.CommandError(msg)
    if parsed_args.wait:
        s = heat_client.stacks.get(stack)
        stack_status, msg = event_utils.poll_for_events(heat_client, s.stack_name, action=action_name, marker=marker)
        if action_name:
            if stack_status == '%s_FAILED' % action_name:
                raise exc.CommandError(msg)
        elif stack_status.endswith('_FAILED'):
            raise exc.CommandError(msg)
    return heat_client.stacks.get(stack)
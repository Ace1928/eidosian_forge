import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def translate_filters(params):
    """Translate filter names to their corresponding DB field names.

    :param params: A dictionary containing keys from engine.api.STACK_KEYS
                    and other keys previously leaked to users.
    :returns: A dict containing only valid DB filed names.
    """
    key_map = {rpc_api.STACK_NAME: 'name', rpc_api.STACK_ACTION: 'action', rpc_api.STACK_STATUS: 'status', rpc_api.STACK_STATUS_DATA: 'status_reason', rpc_api.STACK_DISABLE_ROLLBACK: 'disable_rollback', rpc_api.STACK_TIMEOUT: 'timeout', rpc_api.STACK_OWNER: 'username', rpc_api.STACK_PARENT: 'owner_id', rpc_api.STACK_USER_PROJECT_ID: 'stack_user_project_id'}
    for key, field in key_map.items():
        value = params.pop(key, None)
        if not value:
            continue
        fld_value = params.get(field, None)
        if fld_value:
            if not isinstance(fld_value, list):
                fld_value = [fld_value]
            if not isinstance(value, list):
                value = [value]
            value.extend(fld_value)
        params[field] = value
    if 'status' in params:
        a_set, s_set = _parse_object_status(params['status'])
        statuses = sorted(s_set)
        params['status'] = statuses[0] if len(statuses) == 1 else statuses
        if a_set:
            a = params.get('action', [])
            action_set = set(a) if isinstance(a, list) else set([a])
            actions = sorted(action_set.union(a_set))
            params['action'] = actions[0] if len(actions) == 1 else actions
    return params
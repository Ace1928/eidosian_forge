from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_http_host_condition(self, action, item):
    options = ['host_begins_with_any', 'host_begins_not_with_any', 'host_ends_with_any', 'host_ends_not_with_any', 'host_is_any', 'host_is_not_any', 'host_contains']
    action['type'] = 'http_host'
    if not any((x for x in options if x in item)):
        raise F5ModuleError("A 'host_begins_with_any', 'host_begins_not_with_any', 'host_ends_with_any', 'host_ends_not_with_any','host_is_any', 'host_contains, or 'host_is_not_any' must be specified when the 'http_uri' type is used.")
    if 'host_begins_with_any' in item and item['host_begins_with_any'] is not None:
        if isinstance(item['host_begins_with_any'], list):
            values = item['host_begins_with_any']
        else:
            values = [item['host_begins_with_any']]
        action.update(dict(host=True, startsWith=True, values=values))
    elif 'host_begins_not_with_any' in item and item['host_begins_not_with_any'] is not None:
        if isinstance(item['host_begins_not_with_any'], list):
            values = item['host_begins_not_with_any']
        else:
            values = [item['host_begins_not_with_any']]
        action.update({'host': True, 'startsWith': True, 'not': True, 'values': values})
    elif 'host_ends_not_with_any' in item and item['host_ends_not_with_any'] is not None:
        if isinstance(item['host_ends_not_with_any'], list):
            values = item['host_ends_not_with_any']
        else:
            values = [item['host_ends_not_with_any']]
        action.update({'host': True, 'endsWith': True, 'not': True, 'values': values})
    elif 'host_ends_with_any' in item and item['host_ends_with_any'] is not None:
        if isinstance(item['host_ends_with_any'], list):
            values = item['host_ends_with_any']
        else:
            values = [item['host_ends_with_any']]
        action.update(dict(host=True, endsWith=True, values=values))
    elif 'host_is_any' in item and item['host_is_any'] is not None:
        if isinstance(item['host_is_any'], list):
            values = item['host_is_any']
        else:
            values = [item['host_is_any']]
        action.update(dict(equals=True, host=True, values=values))
    elif 'host_is_not_any' in item and item['host_is_not_any'] is not None:
        if isinstance(item['host_is_not_any'], list):
            values = item['host_is_not_any']
        else:
            values = [item['host_is_not_any']]
        action.update({'equals': True, 'host': True, 'not': True, 'values': values})
    elif 'host_contains' in item and item['host_contains'] is not None:
        if isinstance(item['host_contains'], list):
            values = item['host_contains']
        else:
            values = [item['host_contains']]
        action.update(dict(host=True, contains=True, values=values))
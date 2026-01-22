from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_http_uri_condition(self, action, item):
    action['type'] = 'http_uri'
    options = ['path_begins_with_any', 'path_contains', 'path_is_any']
    if all((k not in item for k in options)):
        raise F5ModuleError("A 'path_begins_with_any', 'path_contains' or 'path_is_any' must be specified when the 'http_uri' type is used.")
    if 'path_begins_with_any' in item and item['path_begins_with_any'] is not None:
        if isinstance(item['path_begins_with_any'], list):
            values = item['path_begins_with_any']
        else:
            values = [item['path_begins_with_any']]
        action.update(dict(path=True, startsWith=True, values=values))
    elif 'path_contains' in item and item['path_contains'] is not None:
        if isinstance(item['path_contains'], list):
            values = item['path_contains']
        else:
            values = [item['path_contains']]
        action.update(dict(path=True, contains=True, values=values))
    elif 'path_is_any' in item and item['path_is_any'] is not None:
        if isinstance(item['path_is_any'], list):
            values = item['path_is_any']
        else:
            values = [item['path_is_any']]
        action.update(dict(path=True, equals=True, values=values))
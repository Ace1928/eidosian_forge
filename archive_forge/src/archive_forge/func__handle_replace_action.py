from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_replace_action(self, action, item):
    """Handle the nuances of the replace type

        :param action:
        :param item:
        :return:
        """
    action['type'] = 'replace'
    options = ['http_header', 'http_referer', 'http_host', 'http_connect', 'http_uri']
    if not any((x for x in options if x in item)):
        raise F5ModuleError("A 'http_header', 'http_referer', 'http_host', 'http_connect' or 'http_uri' must be specified when the 'replace' type is used.")
    event_map = dict(client_accepted='clientAccepted', proxy_connect='proxyConnect', proxy_request='proxyRequest', proxy_response='proxyResponse', request='request', response='response', server_connected='serverConnected', ssl_client_hello='sslClientHello')
    type_map = dict(path='path', query_string='queryString', full_string='value')
    if 'http_header' in item and item['http_header']:
        if item['http_header']['value'] is None:
            raise F5ModuleError("The http_header value key is required when action is of type 'replace'.")
        if item['http_header']['event'] is not None:
            action.update({'httpHeader': True, 'tmName': item['http_header']['name'], 'value': item['http_header']['value'], event_map[item['http_header']['event']]: True})
        else:
            action.update({'httpHeader': True, 'tmName': item['http_header']['name'], 'value': item['http_header']['value']})
    if 'http_referer' in item and item['http_referer']:
        if item['http_referer']['value'] is not None:
            action.update({'httpReferer': True, 'value': item['http_referer']['value'], event_map[item['http_referer']['event']]: True})
        else:
            action.update({'httpReferer': True, event_map[item['http_referer']['event']]: True})
    if 'http_connect' in item and item['http_connect']:
        if item['http_connect']['port'] is None:
            action.update({'httpConnect': True, 'host': item['http_connect']['value'], 'port': 0, event_map[item['http_connect']['event']]: True})
        else:
            action.update({'httpConnect': True, 'host': item['http_connect']['value'], 'port': item['http_connect']['port'], event_map[item['http_connect']['event']]: True})
    if 'http_uri' in item and item['http_uri']:
        if item['http_uri']['event'] is not None:
            action.update({'httpUri': True, type_map[item['http_uri']['type']]: item['http_uri']['value'], event_map[item['http_uri']['event']]: True})
        else:
            action.update({'httpUri': True, type_map[item['http_uri']['type']]: item['http_uri']['value']})
    if 'http_host' in item and item['http_host']:
        if item['http_host']['event'] is not None:
            action.update({'httpHost': True, 'value': item['http_host']['value'], event_map[item['http_host']['event']]: True})
        else:
            action.update({'httpHost': True, 'value': item['http_host']['value']})
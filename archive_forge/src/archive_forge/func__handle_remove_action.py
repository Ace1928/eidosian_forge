from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_remove_action(self, action, item):
    """Handle the nuances of the remove type

        :param action:
        :param item:
        :return:
        """
    action['type'] = 'remove'
    options = ['http_header', 'http_referer', 'http_set_cookie', 'http_cookie']
    if not any((x for x in options if x in item)):
        raise F5ModuleError("A 'http_header', 'http_referer', 'http_set_cookie' or 'http_cookie' must be specified when the 'remove' type is used.")
    if 'http_header' in item and item['http_header']:
        if item['http_header']['event'] == 'request':
            action.update(httpHeader=True, tmName=item['http_header']['name'], request=True)
        elif item['http_header']['event'] == 'response':
            action.update(httpHeader=True, tmName=item['http_header']['name'], response=True)
        else:
            action.update(httpHeader=True, tmName=item['http_header']['name'])
    if 'http_referer' in item and item['http_referer']:
        if item['http_referer']['event'] == 'request':
            action.update(httpReferer=True, request=True)
        if item['http_referer']['event'] == 'proxy_connect':
            action.update(httpReferer=True, proxyConnect=True)
        if item['http_referer']['event'] == 'proxy_request':
            action.update(httpReferer=True, proxyRequest=True)
    if 'http_cookie' in item and item['http_cookie']:
        if item['http_cookie']['event'] == 'request':
            action.update(httpCookie=True, tmName=item['http_cookie']['name'], request=True)
        elif item['http_cookie']['event'] == 'proxy_connect':
            action.update(httpCookie=True, tmName=item['http_cookie']['name'], proxyConnect=True)
        elif item['http_cookie']['event'] == 'proxy_request':
            action.update(httpCookie=True, tmName=item['http_cookie']['name'], proxyRequest=True)
        else:
            action.update(httpCookie=True, tmName=item['http_cookie']['name'])
    if 'http_set_cookie' in item and item['http_set_cookie']:
        action.update(httpSetCookie=True, tmName=item['http_set_cookie']['name'], response=True)
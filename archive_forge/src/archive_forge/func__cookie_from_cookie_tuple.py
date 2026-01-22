import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _cookie_from_cookie_tuple(self, tup, request):
    name, value, standard, rest = tup
    domain = standard.get('domain', Absent)
    path = standard.get('path', Absent)
    port = standard.get('port', Absent)
    expires = standard.get('expires', Absent)
    version = standard.get('version', None)
    if version is not None:
        try:
            version = int(version)
        except ValueError:
            return None
    secure = standard.get('secure', False)
    discard = standard.get('discard', False)
    comment = standard.get('comment', None)
    comment_url = standard.get('commenturl', None)
    if path is not Absent and path != '':
        path_specified = True
        path = escape_path(path)
    else:
        path_specified = False
        path = request_path(request)
        i = path.rfind('/')
        if i != -1:
            if version == 0:
                path = path[:i]
            else:
                path = path[:i + 1]
        if len(path) == 0:
            path = '/'
    domain_specified = domain is not Absent
    domain_initial_dot = False
    if domain_specified:
        domain_initial_dot = bool(domain.startswith('.'))
    if domain is Absent:
        req_host, erhn = eff_request_host(request)
        domain = erhn
    elif not domain.startswith('.'):
        domain = '.' + domain
    port_specified = False
    if port is not Absent:
        if port is None:
            port = request_port(request)
        else:
            port_specified = True
            port = re.sub('\\s+', '', port)
    else:
        port = None
    if expires is Absent:
        expires = None
        discard = True
    elif expires <= self._now:
        try:
            self.clear(domain, path, name)
        except KeyError:
            pass
        _debug("Expiring cookie, domain='%s', path='%s', name='%s'", domain, path, name)
        return None
    return Cookie(version, name, value, port, port_specified, domain, domain_specified, domain_initial_dot, path, path_specified, secure, expires, discard, comment, comment_url, rest)
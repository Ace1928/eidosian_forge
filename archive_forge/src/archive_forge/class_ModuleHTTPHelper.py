from __future__ import (absolute_import, division, print_function)
import abc
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url, open_url, NoSSLError, ConnectionError
import ansible.module_utils.six.moves.urllib.error as urllib_error
class ModuleHTTPHelper(HTTPHelper):

    def __init__(self, module):
        self.module = module

    def fetch_url(self, url, method='GET', headers=None, data=None, timeout=None):
        response, info = fetch_url(self.module, url, method=method, headers=headers, data=data, timeout=timeout)
        try:
            if PY3 and response.closed:
                raise TypeError
            content = response.read()
        except (AttributeError, TypeError):
            content = info.pop('body', None)
        return (content, info)
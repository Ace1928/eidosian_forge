from __future__ import (absolute_import, division, print_function)
import json
from ansible.errors import AnsibleParserError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _api_connect(self):
    self.headers = {'User-Agent': 'ansible-icinga2-inv', 'Accept': 'application/json'}
    api_status_url = self.icinga2_url + '/status'
    request_args = {'headers': self.headers, 'url_username': self.icinga2_user, 'url_password': self.icinga2_password, 'validate_certs': self.ssl_verify}
    open_url(api_status_url, **request_args)
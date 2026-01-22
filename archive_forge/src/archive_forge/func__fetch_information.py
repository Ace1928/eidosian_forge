from __future__ import absolute_import, division, print_function
import json
import uuid
import math
import os
import datetime
from copy import deepcopy
from functools import partial
from sys import version as python_version
from threading import Thread
from typing import Iterable
from itertools import chain
from collections import defaultdict
from ipaddress import ip_interface
from ansible.constants import DEFAULT_LOCAL_TMP
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import raise_from
def _fetch_information(self, url):
    results = None
    cache_key = self.get_cache_key(url)
    user_cache_setting = self.get_option('cache')
    attempt_to_read_cache = user_cache_setting and self.use_cache
    if attempt_to_read_cache:
        try:
            results = self._cache[cache_key]
            need_to_fetch = False
        except KeyError:
            need_to_fetch = True
    else:
        need_to_fetch = True
    if need_to_fetch:
        self.display.v('Fetching: ' + url)
        try:
            response = open_url(url, headers=self.headers, timeout=self.timeout, validate_certs=self.validate_certs, follow_redirects=self.follow_redirects, client_cert=self.cert, client_key=self.key, ca_path=self.ca_path)
        except urllib_error.HTTPError as e:
            'This will return the response body when we encounter an error.\n                This is to help determine what might be the issue when encountering an error.\n                Please check issue #294 for more info.\n                '
            if e.code == 403:
                self.display.display('Permission denied: {0}. This may impair functionality of the inventory plugin.'.format(url), color='red')
                return {'results': [], 'next': None}
            raise AnsibleError(to_native(e.fp.read()))
        try:
            raw_data = to_text(response.read(), errors='surrogate_or_strict')
        except UnicodeError:
            raise AnsibleError('Incorrect encoding of fetched payload from NetBox API.')
        try:
            results = self.loader.load(raw_data, json_only=True)
        except ValueError:
            raise AnsibleError('Incorrect JSON payload: %s' % raw_data)
        if user_cache_setting:
            self._cache[cache_key] = results
    return results
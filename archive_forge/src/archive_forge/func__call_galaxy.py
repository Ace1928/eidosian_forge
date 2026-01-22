from __future__ import (absolute_import, division, print_function)
import collections
import datetime
import functools
import hashlib
import json
import os
import stat
import tarfile
import time
import threading
from http import HTTPStatus
from http.client import BadStatusLine, IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import quote as urlquote, urlencode, urlparse, parse_qs, urljoin
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.urls import open_url, prepare_multipart
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash_s
from ansible.utils.path import makedirs_safe
@retry_with_delays_and_condition(backoff_iterator=generate_jittered_backoff(retries=6, delay_base=2, delay_threshold=40), should_retry_error=should_retry_error)
def _call_galaxy(self, url, args=None, headers=None, method=None, auth_required=False, error_context_msg=None, cache=False, cache_key=None):
    url_info = urlparse(url)
    cache_id = get_cache_id(url)
    if not cache_key:
        cache_key = url_info.path
    query = parse_qs(url_info.query)
    if cache and self._cache:
        server_cache = self._cache.setdefault(cache_id, {})
        iso_datetime_format = '%Y-%m-%dT%H:%M:%SZ'
        valid = False
        if cache_key in server_cache:
            expires = datetime.datetime.strptime(server_cache[cache_key]['expires'], iso_datetime_format)
            expires = expires.replace(tzinfo=datetime.timezone.utc)
            valid = datetime.datetime.now(datetime.timezone.utc) < expires
        is_paginated_url = 'page' in query or 'offset' in query
        if valid and (not is_paginated_url):
            path_cache = server_cache[cache_key]
            if path_cache.get('paginated'):
                if '/v3/' in cache_key:
                    res = {'links': {'next': None}}
                else:
                    res = {'next': None}
                res['results'] = []
                for result in path_cache['results']:
                    res['results'].append(result)
            else:
                res = path_cache['results']
            return res
        elif not is_paginated_url:
            expires = datetime.datetime.now(datetime.timezone.utc)
            expires += datetime.timedelta(days=1)
            server_cache[cache_key] = {'expires': expires.strftime(iso_datetime_format), 'paginated': False}
    headers = headers or {}
    self._add_auth_token(headers, url, required=auth_required)
    try:
        display.vvvv('Calling Galaxy at %s' % url)
        resp = open_url(to_native(url), data=args, validate_certs=self.validate_certs, headers=headers, method=method, timeout=self._server_timeout, http_agent=user_agent(), follow_redirects='safe')
    except HTTPError as e:
        raise GalaxyError(e, error_context_msg)
    except Exception as e:
        raise AnsibleError("Unknown error when attempting to call Galaxy at '%s': %s" % (url, to_native(e)), orig_exc=e)
    resp_data = to_text(resp.read(), errors='surrogate_or_strict')
    try:
        data = json.loads(resp_data)
    except ValueError:
        raise AnsibleError("Failed to parse Galaxy response from '%s' as JSON:\n%s" % (resp.url, to_native(resp_data)))
    if cache and self._cache:
        path_cache = self._cache[cache_id][cache_key]
        paginated_key = None
        for key in ['data', 'results']:
            if key in data:
                paginated_key = key
                break
        if paginated_key:
            path_cache['paginated'] = True
            results = path_cache.setdefault('results', [])
            for result in data[paginated_key]:
                results.append(result)
        else:
            path_cache['results'] = data
    return data
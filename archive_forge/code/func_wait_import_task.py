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
@g_connect(['v2', 'v3'])
def wait_import_task(self, task_id, timeout=0):
    """
        Waits until the import process on the Galaxy server has completed or the timeout is reached.

        :param task_id: The id of the import task to wait for. This can be parsed out of the return
            value for GalaxyAPI.publish_collection.
        :param timeout: The timeout in seconds, 0 is no timeout.
        """
    state = 'waiting'
    data = None
    if 'v3' in self.available_api_versions:
        full_url = _urljoin(self.api_server, self.available_api_versions['v3'], 'imports/collections', task_id, '/')
    else:
        full_url = _urljoin(self.api_server, self.available_api_versions['v2'], 'collection-imports', task_id, '/')
    display.display('Waiting until Galaxy import task %s has completed' % full_url)
    start = time.time()
    wait = 2
    while timeout == 0 or time.time() - start < timeout:
        try:
            data = self._call_galaxy(full_url, method='GET', auth_required=True, error_context_msg='Error when getting import task results at %s' % full_url)
        except GalaxyError as e:
            if e.http_code != 404:
                raise
            display.vvv('Galaxy import process has not started, wait %s seconds before trying again' % wait)
            time.sleep(wait)
            continue
        state = data.get('state', 'waiting')
        if data.get('finished_at', None):
            break
        display.vvv('Galaxy import process has a status of %s, wait %d seconds before trying again' % (state, wait))
        time.sleep(wait)
        wait = min(30, wait * 1.5)
    if state == 'waiting':
        raise AnsibleError("Timeout while waiting for the Galaxy import process to finish, check progress at '%s'" % to_native(full_url))
    for message in data.get('messages', []):
        level = message['level']
        if level.lower() == 'error':
            display.error('Galaxy import error message: %s' % message['message'])
        elif level.lower() == 'warning':
            display.warning('Galaxy import warning message: %s' % message['message'])
        else:
            display.vvv('Galaxy import message: %s - %s' % (level, message['message']))
    if state == 'failed':
        code = to_native(data['error'].get('code', 'UNKNOWN'))
        description = to_native(data['error'].get('description', 'Unknown error, see %s for more details' % full_url))
        raise AnsibleError('Galaxy import process failed: %s (Code: %s)' % (description, code))
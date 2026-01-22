import datetime
import functools
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import urllib
import uuid
import requests
import keystoneauth1
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def _http_log_request(self, url, method=None, data=None, json=None, headers=None, query_params=None, logger=None, split_loggers=None):
    string_parts = []
    if self._get_split_loggers(split_loggers):
        logger = utils.get_logger(__name__ + '.request')
    else:
        string_parts.append('REQ:')
        if not logger:
            logger = utils.get_logger(__name__)
    if not logger.isEnabledFor(logging.DEBUG):
        return
    string_parts.append('curl -g -i')
    if self.verify is False:
        string_parts.append('--insecure')
    elif isinstance(self.verify, str):
        string_parts.append('--cacert "%s"' % self.verify)
    if method:
        string_parts.extend(['-X', method])
    if query_params:
        url = url + '?' + urllib.parse.urlencode(query_params)
        string_parts.append('"%s"' % url)
    else:
        string_parts.append(url)
    if headers:
        for header in sorted(headers.items()):
            string_parts.append('-H "%s: %s"' % self._process_header(header))
    if json:
        data = self._json.encode(json)
    if data:
        if isinstance(data, bytes):
            try:
                data = data.decode('ascii')
            except UnicodeDecodeError:
                data = '<binary_data>'
        string_parts.append("-d '%s'" % data)
    logger.debug(' '.join(string_parts))
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
def _http_log_response(self, response=None, json=None, status_code=None, headers=None, text=None, logger=None, split_loggers=True):
    string_parts = []
    body_parts = []
    if self._get_split_loggers(split_loggers):
        logger = utils.get_logger(__name__ + '.response')
        body_logger = utils.get_logger(__name__ + '.body')
    else:
        string_parts.append('RESP:')
        body_parts.append('RESP BODY:')
        body_logger = logger
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if response is not None:
        if not status_code:
            status_code = response.status_code
        if not headers:
            headers = response.headers
    if status_code:
        string_parts.append('[%s]' % status_code)
    if headers:
        for header in sorted(headers.items()):
            string_parts.append('%s: %s' % self._process_header(header))
    logger.debug(' '.join(string_parts))
    if not body_logger.isEnabledFor(logging.DEBUG):
        return
    if response is not None:
        if not text:
            content_type = response.headers.get('content-type', None)
            for log_type in _LOG_CONTENT_TYPES:
                if content_type is not None and content_type.startswith(log_type):
                    text = self._remove_service_catalog(response.text)
                    break
            else:
                text = 'Omitted, Content-Type is set to %s. Only %s responses have their bodies logged.'
                text = text % (content_type, ', '.join(_LOG_CONTENT_TYPES))
    if json:
        text = self._json.encode(json)
    if text:
        body_parts.append(text)
        body_logger.debug(' '.join(body_parts))
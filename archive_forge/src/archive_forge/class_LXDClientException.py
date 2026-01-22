from __future__ import (absolute_import, division, print_function)
import os
import socket
import ssl
import json
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.common.text.converters import to_text
class LXDClientException(Exception):

    def __init__(self, msg, **kwargs):
        self.msg = msg
        self.kwargs = kwargs
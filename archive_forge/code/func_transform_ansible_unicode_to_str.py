from __future__ import (absolute_import, division, print_function)
import getpass
import os
import socket
import sys
import time
import uuid
from collections import OrderedDict
from os.path import basename
from ansible.errors import AnsibleError
from ansible.module_utils.six import raise_from
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.plugins.callback import CallbackBase
@staticmethod
def transform_ansible_unicode_to_str(value):
    parsed_url = urlparse(str(value))
    if OpenTelemetrySource.is_valid_url(parsed_url):
        return OpenTelemetrySource.redact_user_password(parsed_url).geturl()
    return str(value)
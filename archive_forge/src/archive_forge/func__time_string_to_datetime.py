from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@staticmethod
def _time_string_to_datetime(time_string):
    result = None
    if time_string == 'always':
        result = _ALWAYS
    elif time_string == 'forever':
        result = _FOREVER
    elif is_relative_time_string(time_string):
        result = convert_relative_to_datetime(time_string)
    else:
        for time_format in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
            try:
                result = datetime.strptime(time_string, time_format)
            except ValueError:
                pass
        if result is None:
            raise ValueError
    return result
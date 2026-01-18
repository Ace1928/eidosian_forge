from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def write_debug_log_to_file(self, tag=None, filepath=None, append=True):
    if tag is None:
        tag = 'Debug'
    for status_code, message in self.debug_logs:
        self.write_to_file(tag, status_code, filepath, append)
        if not append:
            append = True
        self.write_to_file(tag, message, filepath, append)
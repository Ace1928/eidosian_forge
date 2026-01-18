from __future__ import absolute_import, division, print_function
import json
import socket
import getpass
from datetime import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.urls import open_url
from ansible.plugins.callback import CallbackBase
def to_millis(dt):
    return int(dt.strftime('%s')) * 1000
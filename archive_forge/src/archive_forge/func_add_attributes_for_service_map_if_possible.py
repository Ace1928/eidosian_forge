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
def add_attributes_for_service_map_if_possible(self, span, task_data):
    """Update the span attributes with the service that the task interacted with, if possible."""
    redacted_url = self.parse_and_redact_url_if_possible(task_data.args)
    if redacted_url:
        span.set_attribute('http.url', redacted_url.geturl())
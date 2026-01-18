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
def traceparent_context(self, traceparent):
    carrier = dict()
    carrier['traceparent'] = traceparent
    return TraceContextTextMapPropagator().extract(carrier=carrier)
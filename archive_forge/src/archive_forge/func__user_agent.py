import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
def _user_agent(self):
    user_agent_suffix = ' '.join(['(%s)' % x for x in self.ua])
    if self.driver:
        user_agent = 'libcloud/{} ({}) {}'.format(libcloud.__version__, self.driver.name, user_agent_suffix)
    else:
        user_agent = 'libcloud/{} {}'.format(libcloud.__version__, user_agent_suffix)
    return user_agent
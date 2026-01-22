from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def add_user_agent(self, config):
    config.add_user_agent(ANSIBLE_USER_AGENT)
    if CLOUDSHELL_USER_AGENT_KEY in os.environ:
        config.add_user_agent(os.environ[CLOUDSHELL_USER_AGENT_KEY])
    if VSCODEEXT_USER_AGENT_KEY in os.environ:
        config.add_user_agent(os.environ[VSCODEEXT_USER_AGENT_KEY])
    return config
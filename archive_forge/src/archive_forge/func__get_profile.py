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
def _get_profile(self, profile='default'):
    path = expanduser('~/.azure/credentials')
    try:
        config = configparser.ConfigParser()
        config.read(path)
    except Exception as exc:
        self.fail('Failed to access {0}. Check that the file exists and you have read access. {1}'.format(path, str(exc)))
    credentials = dict()
    for key in AZURE_CREDENTIAL_ENV_MAPPING:
        try:
            credentials[key] = config.get(profile, key, raw=True)
        except Exception:
            pass
    if credentials.get('subscription_id'):
        return credentials
    return None
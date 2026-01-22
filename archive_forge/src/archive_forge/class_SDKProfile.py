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
class SDKProfile(object):

    def __init__(self, default_api_version, profile=None):
        """Constructor.

        :param str default_api_version: Default API version if not overridden by a profile. Nullable.
        :param profile: A dict operation group name to API version.
        :type profile: dict[str, str]
        """
        self.profile = profile if profile is not None else {}
        self.profile[None] = default_api_version

    @property
    def default_api_version(self):
        return self.profile[None]
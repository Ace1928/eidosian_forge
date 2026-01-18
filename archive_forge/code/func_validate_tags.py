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
def validate_tags(self, tags):
    """
        Check if tags dictionary contains string:string pairs.

        :param tags: dictionary of string:string pairs
        :return: None
        """
    if not self.facts_module:
        if not isinstance(tags, dict):
            self.fail('Tags must be a dictionary of string:string values.')
        for key, value in tags.items():
            if not isinstance(value, str):
                self.fail('Tags values must be strings. Found {0}:{1}'.format(str(key), str(value)))
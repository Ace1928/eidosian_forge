from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _map_reset_type(self, reset_type, allowable_values):
    equiv_types = {'On': 'ForceOn', 'ForceOn': 'On', 'ForceOff': 'GracefulShutdown', 'GracefulShutdown': 'ForceOff', 'GracefulRestart': 'ForceRestart', 'ForceRestart': 'GracefulRestart'}
    if reset_type in allowable_values:
        return reset_type
    if reset_type not in equiv_types:
        return reset_type
    mapped_type = equiv_types[reset_type]
    if mapped_type in allowable_values:
        return mapped_type
    return reset_type
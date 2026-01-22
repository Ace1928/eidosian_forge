from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def aci_owner_spec():
    return dict(owner_key=dict(type='str', no_log=False, fallback=(env_fallback, ['ACI_OWNER_KEY'])), owner_tag=dict(type='str', fallback=(env_fallback, ['ACI_OWNER_TAG'])))
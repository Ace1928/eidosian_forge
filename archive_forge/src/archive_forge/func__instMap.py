from __future__ import absolute_import, division, print_function
import json
import hashlib
import hmac
import locale
from time import strftime, gmtime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import string_types
def _instMap(self, type):
    map = {}
    results = {}
    for result in getattr(self, 'get' + type.title() + 's')():
        map[result['name']] = result['id']
        results[result['id']] = result
    setattr(self, type + '_map', map)
    setattr(self, type + 's', results)
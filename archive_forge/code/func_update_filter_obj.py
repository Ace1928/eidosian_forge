from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def update_filter_obj(self, contract_obj, filter_obj, filter_type, contract_display_name=None, update_filter_ref=True):
    """update filter with more information"""
    if update_filter_ref:
        filter_obj['filterRef'] = self.dict_from_ref(filter_obj.get('filterRef'))
    if contract_display_name:
        filter_obj['displayName'] = contract_display_name
    else:
        filter_obj['displayName'] = contract_obj.get('displayName')
    filter_obj['filterType'] = filter_type
    filter_obj['contractScope'] = contract_obj.get('scope')
    filter_obj['contractFilterType'] = contract_obj.get('filterType')
    if contract_obj.get('description') or contract_obj.get('description') == '':
        filter_obj['description'] = contract_obj.get('description')
    if contract_obj.get('prio'):
        filter_obj['prio'] = contract_obj.get('prio')
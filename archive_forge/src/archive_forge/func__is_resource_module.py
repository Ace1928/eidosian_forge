from __future__ import absolute_import, division, print_function
import copy
import glob
import os
from importlib import import_module
from ansible.errors import AnsibleActionFail, AnsibleError
from ansible.module_utils._text import to_text
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.action.network import (
def _is_resource_module(self, docs):
    doc_obj = yaml.load(docs, SafeLoader)
    if 'config' in doc_obj['options'] and 'state' in doc_obj['options']:
        return True
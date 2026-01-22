from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils import six
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
@staticmethod
def _term_to_secret_id(term):
    try:
        return int(term)
    except ValueError:
        raise AnsibleOptionsError('Secret ID must be an integer')
from __future__ import absolute_import, division, print_function
import os
from importlib import import_module
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import to_list
def _get_sub_plugin_options(self, name):
    return self._sub_plugin_options.get(name)
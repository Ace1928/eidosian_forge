from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
def _stderr_failed(self, err):
    errors_regex = '^sysctl: setting key "[^"]+": (Invalid argument|Read-only file system)$'
    return re.search(errors_regex, err, re.MULTILINE) is not None
from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
@classmethod
def valid_brew_path(cls, brew_path):
    """
        `brew_path` must be one of:
         - None
         - a string containing only:
             - alphanumeric characters
             - dashes
             - dots
             - spaces
             - os.path.sep
        """
    if brew_path is None:
        return True
    return isinstance(brew_path, string_types) and (not cls.INVALID_BREW_PATH_REGEX.search(brew_path))
from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
@classmethod
def valid_path(cls, path):
    """
        `path` must be one of:
         - list of paths
         - a string containing only:
             - alphanumeric characters
             - dashes
             - dots
             - spaces
             - colons
             - os.path.sep
        """
    if isinstance(path, string_types):
        return not cls.INVALID_PATH_REGEX.search(path)
    try:
        iter(path)
    except TypeError:
        return False
    else:
        paths = path
        return all((cls.valid_brew_path(path_) for path_ in paths))
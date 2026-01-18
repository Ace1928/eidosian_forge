from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
@classmethod
def valid_cask(cls, cask):
    """A valid cask is either None or alphanumeric + backslashes."""
    if cask is None:
        return True
    return isinstance(cask, string_types) and (not cls.INVALID_CASK_REGEX.search(cask))
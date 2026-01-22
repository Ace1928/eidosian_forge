from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass
def _require_finder(cls):
    if not cls._collection_finder:
        raise NotImplementedError('an AnsibleCollectionFinder has not been installed in this process')
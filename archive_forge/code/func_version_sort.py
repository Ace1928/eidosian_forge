from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def version_sort(value, reverse=False):
    """Sort a list according to loose versions so that e.g. 2.9 is smaller than 2.10"""
    return sorted(value, key=LooseVersion, reverse=reverse)
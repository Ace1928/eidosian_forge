from __future__ import (absolute_import, division, print_function)
import os
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def _lsb_release_file(self, etc_lsb_release_location):
    lsb_facts = {}
    if not os.path.exists(etc_lsb_release_location):
        return lsb_facts
    for line in get_file_lines(etc_lsb_release_location):
        value = line.split('=', 1)[1].strip()
        if 'DISTRIB_ID' in line:
            lsb_facts['id'] = value
        elif 'DISTRIB_RELEASE' in line:
            lsb_facts['release'] = value
        elif 'DISTRIB_DESCRIPTION' in line:
            lsb_facts['description'] = value
        elif 'DISTRIB_CODENAME' in line:
            lsb_facts['codename'] = value
    return lsb_facts
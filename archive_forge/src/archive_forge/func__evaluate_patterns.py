from __future__ import (absolute_import, division, print_function)
import fnmatch
import os
import sys
import re
import itertools
import traceback
from operator import attrgetter
from random import shuffle
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.data import InventoryData
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins.loader import inventory_loader
from ansible.utils.helpers import deduplicate_list
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
from ansible.vars.plugins import get_vars_from_inventory_sources
def _evaluate_patterns(self, patterns):
    """
        Takes a list of patterns and returns a list of matching host names,
        taking into account any negative and intersection patterns.
        """
    patterns = order_patterns(patterns)
    hosts = []
    for p in patterns:
        if p in self._inventory.hosts:
            hosts.append(self._inventory.get_host(p))
        else:
            that = self._match_one_pattern(p)
            if p[0] == '!':
                that = set(that)
                hosts = [h for h in hosts if h not in that]
            elif p[0] == '&':
                that = set(that)
                hosts = [h for h in hosts if h in that]
            else:
                existing_hosts = set((y.name for y in hosts))
                hosts.extend([h for h in that if h.name not in existing_hosts])
    return hosts
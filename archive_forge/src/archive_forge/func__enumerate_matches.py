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
def _enumerate_matches(self, pattern):
    """
        Returns a list of host names matching the given pattern according to the
        rules explained above in _match_one_pattern.
        """
    results = []
    matching_groups = self._match_list(self._inventory.groups, pattern)
    if matching_groups:
        for groupname in matching_groups:
            results.extend(self._inventory.groups[groupname].get_hosts())
    if not matching_groups or pattern[0] == '~' or any((special in pattern for special in ('.', '?', '*', '['))):
        matching_hosts = self._match_list(self._inventory.hosts, pattern)
        if matching_hosts:
            for hostname in matching_hosts:
                results.append(self._inventory.hosts[hostname])
    if not results and pattern in C.LOCALHOST:
        implicit = self._inventory.get_host(pattern)
        if implicit:
            results.append(implicit)
    if not results and (not matching_groups) and (pattern != 'all'):
        msg = 'Could not match supplied host pattern, ignoring: %s' % pattern
        display.debug(msg)
        if C.HOST_PATTERN_MISMATCH == 'warning':
            display.warning(msg)
        elif C.HOST_PATTERN_MISMATCH == 'error':
            raise AnsibleError(msg)
    return results
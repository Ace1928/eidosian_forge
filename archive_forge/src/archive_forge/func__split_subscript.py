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
def _split_subscript(self, pattern):
    """
        Takes a pattern, checks if it has a subscript, and returns the pattern
        without the subscript and a (start,end) tuple representing the given
        subscript (or None if there is no subscript).

        Validates that the subscript is in the right syntax, but doesn't make
        sure the actual indices make sense in context.
        """
    if pattern[0] == '~':
        return (pattern, None)
    subscript = None
    m = PATTERN_WITH_SUBSCRIPT.match(pattern)
    if m:
        pattern, idx, start, sep, end = m.groups()
        if idx:
            subscript = (int(idx), None)
        else:
            if not end:
                end = -1
            subscript = (int(start), int(end))
            if sep == '-':
                display.warning('Use [x:y] inclusive subscripts instead of [x-y] which has been removed')
    return (pattern, subscript)
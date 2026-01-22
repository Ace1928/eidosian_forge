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
def _fetch_inventory_plugins(self):
    """ sets up loaded inventory plugins for usage """
    display.vvvv('setting up inventory plugins')
    plugins = []
    for name in C.INVENTORY_ENABLED:
        plugin = inventory_loader.get(name)
        if plugin:
            plugins.append(plugin)
        else:
            display.warning('Failed to load inventory plugin, skipping %s' % name)
    if not plugins:
        raise AnsibleError('No inventory plugins available to generate inventory, make sure you have at least one enabled.')
    return plugins
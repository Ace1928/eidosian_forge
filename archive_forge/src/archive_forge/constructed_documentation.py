from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible.errors import AnsibleParserError, AnsibleOptionsError
from ansible.inventory.helpers import get_group_vars
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.common.text.converters import to_native
from ansible.utils.vars import combine_vars
from ansible.vars.fact_cache import FactCache
from ansible.vars.plugins import get_vars_from_inventory_sources
 parses the inventory file 
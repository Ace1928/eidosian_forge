from __future__ import (absolute_import, division, print_function)
import os
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.vars import BaseVarsPlugin
from ansible.utils.path import basedir
from ansible.inventory.group import InventoryObjectType
from ansible.utils.vars import combine_vars
 parses the inventory file 
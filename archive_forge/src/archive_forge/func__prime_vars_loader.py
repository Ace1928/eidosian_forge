from __future__ import annotations
import os
from functools import lru_cache
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.inventory.group import InventoryObjectType
from ansible.plugins.loader import vars_loader
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def _prime_vars_loader():
    list(vars_loader.all(class_only=True))
    for plugin_name in C.VARIABLE_PLUGINS_ENABLED:
        if not plugin_name:
            continue
        vars_loader.get(plugin_name)
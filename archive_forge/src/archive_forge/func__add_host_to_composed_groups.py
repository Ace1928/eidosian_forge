from __future__ import (absolute_import, division, print_function)
import hashlib
import os
import string
from collections.abc import Mapping
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.inventory.group import to_safe_group_name as original_safe
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins import AnsiblePlugin
from ansible.plugins.cache import CachePluginAdjudicator as CacheObject
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, load_extra_vars
def _add_host_to_composed_groups(self, groups, variables, host, strict=False, fetch_hostvars=True):
    """ helper to create complex groups for plugins based on jinja2 conditionals, hosts that meet the conditional are added to group"""
    if groups and isinstance(groups, dict):
        if fetch_hostvars:
            variables = combine_vars(variables, self.inventory.get_host(host).get_vars())
        self.templar.available_variables = variables
        for group_name in groups:
            conditional = '{%% if %s %%} True {%% else %%} False {%% endif %%}' % groups[group_name]
            group_name = self._sanitize_group_name(group_name)
            try:
                result = boolean(self.templar.template(conditional))
            except Exception as e:
                if strict:
                    raise AnsibleParserError('Could not add host %s to group %s: %s' % (host, group_name, to_native(e)))
                continue
            if result:
                group_name = self.inventory.add_group(group_name)
                self.inventory.add_child(group_name, host)
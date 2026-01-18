from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import sys
import argparse
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.utils.vars import combine_vars
from ansible.utils.display import Display
from ansible.vars.plugins import get_vars_from_inventory_sources, get_vars_from_path
def yaml_inventory(self, top):
    seen_hosts = set()
    seen_groups = set()

    def format_group(group, available_hosts):
        results = {}
        results[group.name] = {}
        results[group.name]['children'] = {}
        for subgroup in group.child_groups:
            if subgroup.name != 'all':
                if subgroup.name in seen_groups:
                    results[group.name]['children'].update({subgroup.name: {}})
                else:
                    results[group.name]['children'].update(format_group(subgroup, available_hosts))
                    seen_groups.add(subgroup.name)
        results[group.name]['hosts'] = {}
        if group.name != 'all':
            for h in group.hosts:
                if h.name not in available_hosts:
                    continue
                myvars = {}
                if h.name not in seen_hosts:
                    seen_hosts.add(h.name)
                    myvars = self._get_host_variables(host=h)
                results[group.name]['hosts'][h.name] = myvars
        if context.CLIARGS['export']:
            gvars = self._get_group_variables(group)
            if gvars:
                results[group.name]['vars'] = gvars
        self._remove_empty_keys(results[group.name])
        if not results[group.name]:
            del results[group.name]
        return results
    available_hosts = frozenset((h.name for h in self.inventory.get_hosts(top.name)))
    return format_group(top, available_hosts)
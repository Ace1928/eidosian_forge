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
def _get_group_variables(self, group):
    res = group.get_vars()
    res = combine_vars(res, get_vars_from_inventory_sources(self.loader, self.inventory._sources, [group], 'all'))
    if context.CLIARGS['basedir']:
        res = combine_vars(res, get_vars_from_path(self.loader, context.CLIARGS['basedir'], [group], 'all'))
    if group.priority != 1:
        res['ansible_group_priority'] = group.priority
    return self._remove_internal(res)
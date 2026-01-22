from __future__ import (absolute_import, division, print_function)
import os
import pty
import time
import json
import signal
import subprocess
import sys
import termios
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleConnectionFailure, AnsibleActionFail, AnsibleActionSkip
from ansible.executor.task_result import TaskResult
from ansible.executor.module_common import get_action_args_with_defaults
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.connection import write_to_file_descriptor
from ansible.playbook.conditional import Conditional
from ansible.playbook.task import Task
from ansible.plugins import get_plugin_class
from ansible.plugins.loader import become_loader, cliconf_loader, connection_loader, httpapi_loader, netconf_loader, terminal_loader
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.unsafe_proxy import to_unsafe_text, wrap_var
from ansible.vars.clean import namespace_facts, clean_facts
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, isidentifier
def _set_connection_options(self, variables, templar):
    varnames = []
    option_vars = C.config.get_plugin_vars('connection', self._connection._load_name)
    varnames.extend(option_vars)
    options = {'_extras': {}}
    for k in option_vars:
        if k in variables:
            options[k] = templar.template(variables[k])
    if getattr(self._connection, 'allow_extras', False):
        for k in variables:
            if k.startswith('ansible_%s_' % self._connection._load_name) and k not in options:
                options['_extras'][k] = templar.template(variables[k])
    task_keys = self._task.dump_attrs()
    task_keys['timeout'] = self._play_context.timeout
    if self._play_context.password:
        task_keys['password'] = self._play_context.password
    del task_keys['retries']
    self._connection.set_options(task_keys=task_keys, var_options=options)
    varnames.extend(self._set_plugin_options('shell', variables, templar, task_keys))
    if self._connection.become is not None:
        if self._play_context.become_pass:
            task_keys['become_pass'] = self._play_context.become_pass
        varnames.extend(self._set_plugin_options('become', variables, templar, task_keys))
        for option in ('become_user', 'become_flags', 'become_exe', 'become_pass'):
            try:
                setattr(self._play_context, option, self._connection.become.get_option(option))
            except KeyError:
                pass
        self._play_context.prompt = self._connection.become.prompt
    sub = getattr(self._connection, '_sub_plugin', None)
    if sub and sub.get('type') != 'external':
        plugin_type = get_plugin_class(sub.get('obj'))
        varnames.extend(self._set_plugin_options(plugin_type, variables, templar, task_keys))
    sub_conn = getattr(self._connection, 'ssh_type_conn', None)
    if sub_conn is not None:
        varnames.extend(self._set_plugin_options('ssh_type_conn', variables, templar, task_keys))
    return varnames
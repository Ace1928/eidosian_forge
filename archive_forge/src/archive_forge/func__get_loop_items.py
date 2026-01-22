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
def _get_loop_items(self):
    """
        Loads a lookup plugin to handle the with_* portion of a task (if specified),
        and returns the items result.
        """
    self._job_vars['ansible_search_path'] = self._task.get_search_path()
    if self._loader.get_basedir() not in self._job_vars['ansible_search_path']:
        self._job_vars['ansible_search_path'].append(self._loader.get_basedir())
    templar = Templar(loader=self._loader, variables=self._job_vars)
    items = None
    if self._task.loop_with:
        if self._task.loop_with in self._shared_loader_obj.lookup_loader:
            fail = bool(self._task.loop_with != 'first_found')
            loop_terms = listify_lookup_plugin_terms(terms=self._task.loop, templar=templar, fail_on_undefined=fail, convert_bare=False)
            mylookup = self._shared_loader_obj.lookup_loader.get(self._task.loop_with, loader=self._loader, templar=templar)
            for subdir in ['template', 'var', 'file']:
                if subdir in self._task.action:
                    break
            setattr(mylookup, '_subdir', subdir + 's')
            items = wrap_var(mylookup.run(terms=loop_terms, variables=self._job_vars, wantlist=True))
        else:
            raise AnsibleError("Unexpected failure in finding the lookup named '%s' in the available lookup plugins" % self._task.loop_with)
    elif self._task.loop is not None:
        items = templar.template(self._task.loop)
        if not isinstance(items, list):
            raise AnsibleError("Invalid data passed to 'loop', it requires a list, got this instead: %s. Hint: If you passed a list/dict of just one element, try adding wantlist=True to your lookup invocation or use q/query instead of lookup." % items)
    return items
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
def _run_loop(self, items):
    """
        Runs the task with the loop items specified and collates the result
        into an array named 'results' which is inserted into the final result
        along with the item for which the loop ran.
        """
    task_vars = self._job_vars
    templar = Templar(loader=self._loader, variables=task_vars)
    self._task.loop_control.post_validate(templar=templar)
    loop_var = self._task.loop_control.loop_var
    index_var = self._task.loop_control.index_var
    loop_pause = self._task.loop_control.pause
    extended = self._task.loop_control.extended
    extended_allitems = self._task.loop_control.extended_allitems
    label = self._task.loop_control.label or '{{' + loop_var + '}}'
    if loop_var in task_vars:
        display.warning(u"%s: The loop variable '%s' is already in use. You should set the `loop_var` value in the `loop_control` option for the task to something else to avoid variable collisions and unexpected behavior." % (self._task, loop_var))
    ran_once = False
    task_fields = None
    no_log = False
    items_len = len(items)
    results = []
    for item_index, item in enumerate(items):
        task_vars['ansible_loop_var'] = loop_var
        task_vars[loop_var] = item
        if index_var:
            task_vars['ansible_index_var'] = index_var
            task_vars[index_var] = item_index
        if extended:
            task_vars['ansible_loop'] = {'index': item_index + 1, 'index0': item_index, 'first': item_index == 0, 'last': item_index + 1 == items_len, 'length': items_len, 'revindex': items_len - item_index, 'revindex0': items_len - item_index - 1}
            if extended_allitems:
                task_vars['ansible_loop']['allitems'] = items
            try:
                task_vars['ansible_loop']['nextitem'] = items[item_index + 1]
            except IndexError:
                pass
            if item_index - 1 >= 0:
                task_vars['ansible_loop']['previtem'] = items[item_index - 1]
        templar.available_variables = task_vars
        if loop_pause and ran_once:
            time.sleep(loop_pause)
        else:
            ran_once = True
        try:
            tmp_task = self._task.copy(exclude_parent=True, exclude_tasks=True)
            tmp_task._parent = self._task._parent
            tmp_play_context = self._play_context.copy()
        except AnsibleParserError as e:
            results.append(dict(failed=True, msg=to_text(e)))
            continue
        self._task, tmp_task = (tmp_task, self._task)
        self._play_context, tmp_play_context = (tmp_play_context, self._play_context)
        res = self._execute(variables=task_vars)
        task_fields = self._task.dump_attrs()
        self._task, tmp_task = (tmp_task, self._task)
        self._play_context, tmp_play_context = (tmp_play_context, self._play_context)
        no_log = no_log or tmp_task.no_log
        res[loop_var] = item
        res['ansible_loop_var'] = loop_var
        if index_var:
            res[index_var] = item_index
            res['ansible_index_var'] = index_var
        if extended:
            res['ansible_loop'] = task_vars['ansible_loop']
        res['_ansible_item_result'] = True
        res['_ansible_ignore_errors'] = task_fields.get('ignore_errors')
        res['_ansible_ignore_unreachable'] = task_fields.get('ignore_unreachable')
        try:
            res['_ansible_item_label'] = templar.template(label)
        except AnsibleUndefinedVariable as e:
            res.update({'failed': True, 'msg': 'Failed to template loop_control.label: %s' % to_text(e)})
        tr = TaskResult(self._host.name, self._task._uuid, res, task_fields=task_fields)
        if tr.is_failed() or tr.is_unreachable():
            self._final_q.send_callback('v2_runner_item_on_failed', tr)
        elif tr.is_skipped():
            self._final_q.send_callback('v2_runner_item_on_skipped', tr)
        else:
            if getattr(self._task, 'diff', False):
                self._final_q.send_callback('v2_on_file_diff', tr)
            if self._task.action not in C._ACTION_INVENTORY_TASKS:
                self._final_q.send_callback('v2_runner_item_on_ok', tr)
        results.append(res)
        del task_vars[loop_var]
        if self._connection:
            clear_plugins = {'connection': self._connection._load_name, 'shell': self._connection._shell._load_name}
            if self._connection.become:
                clear_plugins['become'] = self._connection.become._load_name
            for plugin_type, plugin_name in clear_plugins.items():
                for var in C.config.get_plugin_vars(plugin_type, plugin_name):
                    if var in task_vars and var not in self._job_vars:
                        del task_vars[var]
    self._task.no_log = no_log
    self._task.run_once = task_fields.get('run_once')
    self._task.action = task_fields.get('action')
    return results
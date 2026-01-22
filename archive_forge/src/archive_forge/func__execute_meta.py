from __future__ import (absolute_import, division, print_function)
import cmd
import functools
import os
import pprint
import queue
import sys
import threading
import time
import typing as t
from collections import deque
from multiprocessing import Lock
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleUndefinedVariable, AnsibleParserError
from ansible.executor import action_write_locks
from ansible.executor.play_iterator import IteratingStates, PlayIterator
from ansible.executor.process.worker import WorkerProcess
from ansible.executor.task_result import TaskResult
from ansible.executor.task_queue_manager import CallbackSend, DisplaySend, PromptSend
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.playbook.conditional import Conditional
from ansible.playbook.handler import Handler
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.task import Task
from ansible.playbook.task_include import TaskInclude
from ansible.plugins import loader as plugin_loader
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.fqcn import add_internal_fqcns
from ansible.utils.unsafe_proxy import wrap_var
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier
from ansible.vars.clean import strip_internal_keys, module_response_deepcopy
def _execute_meta(self, task, play_context, iterator, target_host):
    meta_action = task.args.get('_raw_params')

    def _evaluate_conditional(h):
        all_vars = self._variable_manager.get_vars(play=iterator._play, host=h, task=task, _hosts=self._hosts_cache, _hosts_all=self._hosts_cache_all)
        templar = Templar(loader=self._loader, variables=all_vars)
        return task.evaluate_conditional(templar, all_vars)
    skipped = False
    msg = meta_action
    skip_reason = '%s conditional evaluated to False' % meta_action
    if isinstance(task, Handler):
        self._tqm.send_callback('v2_playbook_on_handler_task_start', task)
    else:
        self._tqm.send_callback('v2_playbook_on_task_start', task, is_conditional=False)
    if meta_action in ('noop', 'refresh_inventory', 'reset_connection') and task.when:
        self._cond_not_supported_warn(meta_action)
    if meta_action == 'noop':
        msg = 'noop'
    elif meta_action == 'flush_handlers':
        if _evaluate_conditional(target_host):
            host_state = iterator.get_state_for_host(target_host.name)
            for notification in list(host_state.handler_notifications):
                for handler in self.search_handlers_by_notification(notification, iterator):
                    if handler.notify_host(target_host):
                        self._tqm.send_callback('v2_playbook_on_notify', handler, target_host)
                iterator.clear_notification(target_host.name, notification)
            if host_state.run_state == IteratingStates.HANDLERS:
                raise AnsibleError('flush_handlers cannot be used as a handler')
            if target_host.name not in self._tqm._unreachable_hosts:
                host_state.pre_flushing_run_state = host_state.run_state
                host_state.run_state = IteratingStates.HANDLERS
            msg = 'triggered running handlers for %s' % target_host.name
        else:
            skipped = True
            skip_reason += ', not running handlers for %s' % target_host.name
    elif meta_action == 'refresh_inventory':
        self._inventory.refresh_inventory()
        self._set_hosts_cache(iterator._play)
        msg = 'inventory successfully refreshed'
    elif meta_action == 'clear_facts':
        if _evaluate_conditional(target_host):
            for host in self._inventory.get_hosts(iterator._play.hosts):
                hostname = host.get_name()
                self._variable_manager.clear_facts(hostname)
            msg = 'facts cleared'
        else:
            skipped = True
            skip_reason += ', not clearing facts and fact cache for %s' % target_host.name
    elif meta_action == 'clear_host_errors':
        if _evaluate_conditional(target_host):
            for host in self._inventory.get_hosts(iterator._play.hosts):
                self._tqm._failed_hosts.pop(host.name, False)
                self._tqm._unreachable_hosts.pop(host.name, False)
                iterator.clear_host_errors(host)
            msg = 'cleared host errors'
        else:
            skipped = True
            skip_reason += ', not clearing host error state for %s' % target_host.name
    elif meta_action == 'end_batch':
        if _evaluate_conditional(target_host):
            for host in self._inventory.get_hosts(iterator._play.hosts):
                if host.name not in self._tqm._unreachable_hosts:
                    iterator.set_run_state_for_host(host.name, IteratingStates.COMPLETE)
            msg = 'ending batch'
        else:
            skipped = True
            skip_reason += ', continuing current batch'
    elif meta_action == 'end_play':
        if _evaluate_conditional(target_host):
            for host in self._inventory.get_hosts(iterator._play.hosts):
                if host.name not in self._tqm._unreachable_hosts:
                    iterator.set_run_state_for_host(host.name, IteratingStates.COMPLETE)
                    iterator.end_play = True
            msg = 'ending play'
        else:
            skipped = True
            skip_reason += ', continuing play'
    elif meta_action == 'end_host':
        if _evaluate_conditional(target_host):
            iterator.set_run_state_for_host(target_host.name, IteratingStates.COMPLETE)
            iterator._play._removed_hosts.append(target_host.name)
            msg = 'ending play for %s' % target_host.name
        else:
            skipped = True
            skip_reason += ', continuing execution for %s' % target_host.name
            msg = 'end_host conditional evaluated to false, continuing execution for %s' % target_host.name
    elif meta_action == 'role_complete':
        if task.implicit:
            role_obj = self._get_cached_role(task, iterator._play)
            if target_host.name in role_obj._had_task_run:
                role_obj._completed[target_host.name] = True
                msg = 'role_complete for %s' % target_host.name
    elif meta_action == 'reset_connection':
        all_vars = self._variable_manager.get_vars(play=iterator._play, host=target_host, task=task, _hosts=self._hosts_cache, _hosts_all=self._hosts_cache_all)
        templar = Templar(loader=self._loader, variables=all_vars)
        play_context = play_context.set_task_and_variable_override(task=task, variables=all_vars, templar=templar)
        play_context.post_validate(templar=templar)
        if not play_context.remote_addr:
            play_context.remote_addr = target_host.address
        play_context.update_vars(all_vars)
        if target_host in self._active_connections:
            connection = Connection(self._active_connections[target_host])
            del self._active_connections[target_host]
        else:
            connection = plugin_loader.connection_loader.get(play_context.connection, play_context, os.devnull)
            connection.set_options(task_keys=task.dump_attrs(), var_options=all_vars)
            play_context.set_attributes_from_plugin(connection)
        if connection:
            try:
                connection.reset()
                msg = 'reset connection'
            except ConnectionError as e:
                display.debug('got an error while closing persistent connection: %s' % e)
        else:
            msg = 'no connection, nothing to reset'
    else:
        raise AnsibleError('invalid meta action requested: %s' % meta_action, obj=task._ds)
    result = {'msg': msg}
    if skipped:
        result['skipped'] = True
        result['skip_reason'] = skip_reason
    else:
        result['changed'] = False
    if not task.implicit:
        header = skip_reason if skipped else msg
        display.vv(f'META: {header}')
    res = TaskResult(target_host, task, result)
    if skipped:
        self._tqm.send_callback('v2_runner_on_skipped', res)
    return [res]
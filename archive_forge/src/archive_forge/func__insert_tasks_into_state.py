from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def _insert_tasks_into_state(self, state, task_list):
    if state.fail_state != FailedStates.NONE and state.run_state == IteratingStates.TASKS or not task_list:
        return state
    if state.run_state == IteratingStates.TASKS:
        if state.tasks_child_state:
            state.tasks_child_state = self._insert_tasks_into_state(state.tasks_child_state, task_list)
        else:
            target_block = state._blocks[state.cur_block].copy()
            before = target_block.block[:state.cur_regular_task]
            after = target_block.block[state.cur_regular_task:]
            target_block.block = before + task_list + after
            state._blocks[state.cur_block] = target_block
    elif state.run_state == IteratingStates.RESCUE:
        if state.rescue_child_state:
            state.rescue_child_state = self._insert_tasks_into_state(state.rescue_child_state, task_list)
        else:
            target_block = state._blocks[state.cur_block].copy()
            before = target_block.rescue[:state.cur_rescue_task]
            after = target_block.rescue[state.cur_rescue_task:]
            target_block.rescue = before + task_list + after
            state._blocks[state.cur_block] = target_block
    elif state.run_state == IteratingStates.ALWAYS:
        if state.always_child_state:
            state.always_child_state = self._insert_tasks_into_state(state.always_child_state, task_list)
        else:
            target_block = state._blocks[state.cur_block].copy()
            before = target_block.always[:state.cur_always_task]
            after = target_block.always[state.cur_always_task:]
            target_block.always = before + task_list + after
            state._blocks[state.cur_block] = target_block
    elif state.run_state == IteratingStates.HANDLERS:
        state.handlers[state.cur_handlers_task:state.cur_handlers_task] = [h for b in task_list for h in b.block]
    return state
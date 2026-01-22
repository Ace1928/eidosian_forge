from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def _set_failed_state(self, state):
    if state.run_state == IteratingStates.SETUP:
        state.fail_state |= FailedStates.SETUP
        state.run_state = IteratingStates.COMPLETE
    elif state.run_state == IteratingStates.TASKS:
        if state.tasks_child_state is not None:
            state.tasks_child_state = self._set_failed_state(state.tasks_child_state)
        else:
            state.fail_state |= FailedStates.TASKS
            if state._blocks[state.cur_block].rescue:
                state.run_state = IteratingStates.RESCUE
            elif state._blocks[state.cur_block].always:
                state.run_state = IteratingStates.ALWAYS
            else:
                state.run_state = IteratingStates.COMPLETE
    elif state.run_state == IteratingStates.RESCUE:
        if state.rescue_child_state is not None:
            state.rescue_child_state = self._set_failed_state(state.rescue_child_state)
        else:
            state.fail_state |= FailedStates.RESCUE
            if state._blocks[state.cur_block].always:
                state.run_state = IteratingStates.ALWAYS
            else:
                state.run_state = IteratingStates.COMPLETE
    elif state.run_state == IteratingStates.ALWAYS:
        if state.always_child_state is not None:
            state.always_child_state = self._set_failed_state(state.always_child_state)
        else:
            state.fail_state |= FailedStates.ALWAYS
            state.run_state = IteratingStates.COMPLETE
    return state
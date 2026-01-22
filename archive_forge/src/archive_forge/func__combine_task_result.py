from __future__ import (absolute_import, division, print_function)
import os
import time
import typing as t
from ansible import constants as C
from ansible.executor.module_common import get_action_args_with_defaults
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.vars import merge_hash
def _combine_task_result(self, result: dict[str, t.Any], task_result: dict[str, t.Any]) -> dict[str, t.Any]:
    filtered_res = {'ansible_facts': task_result.get('ansible_facts', {}), 'warnings': task_result.get('warnings', []), 'deprecations': task_result.get('deprecations', [])}
    return merge_hash(result, filtered_res, list_merge='append_rp')
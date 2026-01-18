import asyncio
import sys
from copy import deepcopy
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass, field
import logging
import numpy as np
import pprint
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple, Union
from ray.util.state import list_tasks
import ray
from ray.actor import ActorHandle
from ray.util.state import list_workers
from ray._private.gcs_utils import GcsAioClient, GcsChannel
from ray.util.state.state_manager import StateDataSourceClient
from ray.dashboard.state_aggregator import (
def verify_tasks_running_or_terminated(task_pids: Dict[str, Tuple[int, Optional[str]]], expect_num_tasks: int):
    """
    Check if the tasks in task_pids are in RUNNING state if pid exists
    and running the task.
    If the pid is missing or the task is not running the task, check if the task
    is marked FAILED or FINISHED.

    Args:
        task_pids: A dict of task name to (pid, expected terminal state).

    """
    import psutil
    assert len(task_pids) == expect_num_tasks, task_pids
    for task_name, pid_and_state in task_pids.items():
        tasks = list_tasks(detail=True, filters=[('name', '=', task_name)])
        assert len(tasks) == 1, f'One unique task with {task_name} should be found. Use `options(name=<task_name>)` when creating the task.'
        task = tasks[0]
        pid, expected_state = pid_and_state
        if sys.platform in ['win32', 'darwin']:
            if expected_state is not None:
                assert task['state'] == expected_state, task
            continue
        if psutil.pid_exists(pid) and task_name in psutil.Process(pid).name():
            assert 'ray::IDLE' not in task['name'], "One should not name it 'IDLE' since it's reserved in Ray"
            assert task['state'] == 'RUNNING', task
            if expected_state is not None:
                assert task['state'] == expected_state, task
        elif expected_state is None:
            assert task['state'] in ['FAILED', 'FINISHED'], f'{task_name}: {task['task_id']} = {task['state']}'
        else:
            assert task['state'] == expected_state, f'expect {expected_state} but {task['state']} for {task}'
    return True
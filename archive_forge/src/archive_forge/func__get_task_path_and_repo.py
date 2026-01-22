from typing import Callable, Dict, Type
import importlib
from collections import namedtuple
def _get_task_path_and_repo(taskname: str):
    """
    Returns the task path list and repository containing the task as specified by
    `--task`.

    :param taskname: path to task class (specified in format detailed below)
    """
    task = taskname.strip()
    repo = 'parlai'
    if task.startswith('internal:'):
        repo = 'parlai_internal'
        task = task[9:]
    elif task.startswith('fb:'):
        repo = 'parlai_fb'
        task = task[3:]
    task_path_list = task.split(':')
    return (task_path_list, repo)
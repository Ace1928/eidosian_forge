import copy
import random
from typing import List, Dict, Union
from parlai.core.agents import create_agents_from_shared
from parlai.core.loader import load_task_module, load_world_module
from parlai.core.metrics import aggregate_named_reports
from parlai.core.opt import Opt
from parlai.core.teachers import Teacher, create_task_agent_from_taskname
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import Timer, display_messages
from parlai.tasks.tasks import ids_to_tasks
import parlai.utils.logging as logging
def _create_task_agents(opt: Opt):
    """
    Create task agent(s) for the given task name.

    It does this by calling the create_agent function in agents.py of the given task. If
    create_agents function does not exist, it just looks for the teacher (agent) class
    defined by the task name directly.  (This saves the task creator bothering to define
    the create_agents function when it is not needed.)
    """
    if opt.get('interactive_task', False) or opt.get('selfchat_task', False):
        return []
    try:
        my_module = load_task_module(opt['task'])
        task_agents = my_module.create_agents(opt)
    except (ModuleNotFoundError, AttributeError):
        return create_task_agent_from_taskname(opt)
    if type(task_agents) != list:
        task_agents = [task_agents]
    return task_agents
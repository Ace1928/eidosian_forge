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
def _override_opts_in_shared(table, overrides):
    """
    Override all shared dicts.

    Looks recursively for ``opt`` dictionaries within shared dict and overrides any key-
    value pairs with pairs from the overrides dict.
    """
    if 'opt' in table:
        for k, v in overrides.items():
            table['opt'][k] = v
    for k, v in table.items():
        if type(v) == dict and k != 'opt' and ('opt' in v):
            _override_opts_in_shared(v, overrides)
        elif type(v) == list:
            for item in v:
                if type(item) == dict and 'opt' in item:
                    _override_opts_in_shared(item, overrides)
                else:
                    break
    return table
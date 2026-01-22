import enum
import os
import pickle
import urllib
import warnings
import numpy as np
from numbers import Number
import pyarrow.fs
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ray
from ray import logger
from ray.air import session
from ray.air._internal import usage as air_usage
from ray.air.util.node import _force_on_current_node
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.tune.experiment import Trial
from ray.train._internal.syncer import DEFAULT_SYNC_TIMEOUT
from ray._private.storage import _load_class
from ray.util import PublicAPI
from ray.util.queue import Queue
def _setup_wandb(trial_id: str, trial_name: str, config: Optional[Dict]=None, api_key: Optional[str]=None, api_key_file: Optional[str]=None, _wandb: Optional[ModuleType]=None, **kwargs) -> Union[Run, RunDisabled]:
    _config = config.copy() if config else {}
    if api_key_file:
        api_key_file = os.path.expanduser(api_key_file)
    _set_api_key(api_key_file, api_key)
    project = _get_wandb_project(kwargs.pop('project', None))
    group = kwargs.pop('group', os.environ.get(WANDB_GROUP_ENV_VAR))
    _config = _clean_log(_config)
    wandb_init_kwargs = dict(id=trial_id, name=trial_name, resume=True, reinit=True, allow_val_change=True, config=_config, project=project, group=group)
    wandb_init_kwargs.update(**kwargs)
    if os.name == 'nt':
        os.environ['WANDB_START_METHOD'] = 'thread'
    else:
        os.environ['WANDB_START_METHOD'] = 'fork'
    _wandb = _wandb or wandb
    run = _wandb.init(**wandb_init_kwargs)
    _run_wandb_process_run_info_hook(run)
    air_usage.tag_setup_wandb()
    return run
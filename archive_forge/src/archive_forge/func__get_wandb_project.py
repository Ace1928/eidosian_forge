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
def _get_wandb_project(project: Optional[str]=None) -> Optional[str]:
    """Get W&B project from environment variable or external hook if not passed
    as and argument."""
    if not project and (not os.environ.get(WANDB_PROJECT_ENV_VAR)) and os.environ.get(WANDB_POPULATE_RUN_LOCATION_HOOK):
        try:
            _load_class(os.environ[WANDB_POPULATE_RUN_LOCATION_HOOK])()
        except Exception as e:
            logger.exception(f'Error executing {WANDB_POPULATE_RUN_LOCATION_HOOK} to populate {WANDB_PROJECT_ENV_VAR} and {WANDB_GROUP_ENV_VAR}: {e}', exc_info=e)
    if not project and os.environ.get(WANDB_PROJECT_ENV_VAR):
        project = os.environ.get(WANDB_PROJECT_ENV_VAR)
    return project
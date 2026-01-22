import json
import os
import random
import string
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import yaml
from wandb import env
from wandb.apis import InternalApi
from wandb.sdk import wandb_sweep
from wandb.sdk.launch.sweeps.utils import (
from wandb.util import get_module
def _sweep_object_sync_to_backend(self) -> None:
    if self._controller == self._controller_prev_step:
        return
    sweep_obj_id = self._sweep_obj['id']
    controller = json.dumps(self._controller)
    _, warnings = self._api.upsert_sweep(self._sweep_config, controller=controller, obj_id=sweep_obj_id)
    handle_sweep_config_violations(warnings)
    self._controller_prev_step = self._controller.copy()
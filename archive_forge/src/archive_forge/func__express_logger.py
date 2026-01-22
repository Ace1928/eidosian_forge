import os
from typing import Any, Dict, Optional
import wandb
from triad import assert_or_throw
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError
from wandb.env import get_project
from wandb.sdk.lib.apikey import api_key
from wandb.wandb_run import Run
@parse_logger.candidate(lambda obj: isinstance(obj, str) and (obj == 'wandb' or obj.startswith('wandb:')))
def _express_logger(obj: str) -> 'WandbGroupLogger':
    p = obj.split(':', 1)
    project_name = p[1] if len(p) > 1 else 'TUNE-DEFAULT'
    return WandbGroupLogger(project_name=project_name)
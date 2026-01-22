import datetime
import io
import json
import re
import time
from typing import Any, Dict, Optional, Tuple
import wandb
from wandb import util
from wandb.data_types import Table
from wandb.sdk.lib import telemetry
from wandb.sdk.wandb_run import Run
from wandb.util import parse_version
from openai import OpenAI  # noqa: E402
from openai.types.fine_tuning import FineTuningJob  # noqa: E402
from openai.types.fine_tuning.fine_tuning_job import Hyperparameters  # noqa: E402
@classmethod
def _unpack_hyperparameters(cls, hyperparameters: Hyperparameters):
    hyperparams = {}
    try:
        hyperparams['n_epochs'] = hyperparameters.n_epochs
        hyperparams['batch_size'] = hyperparameters.batch_size
        hyperparams['learning_rate_multiplier'] = hyperparameters.learning_rate_multiplier
    except Exception:
        return None
    return hyperparams
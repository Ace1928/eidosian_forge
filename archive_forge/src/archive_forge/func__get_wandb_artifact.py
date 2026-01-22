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
def _get_wandb_artifact(cls, artifact_path: str):
    cls._ensure_logged_in()
    try:
        if cls._wandb_api is None:
            cls._wandb_api = wandb.Api()
        return cls._wandb_api.artifact(artifact_path)
    except Exception:
        return None
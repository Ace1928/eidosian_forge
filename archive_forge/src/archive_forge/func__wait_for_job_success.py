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
def _wait_for_job_success(cls, fine_tune: FineTuningJob) -> FineTuningJob:
    wandb.termlog('Waiting for the OpenAI fine-tuning job to be finished...')
    while True:
        if fine_tune.status == 'succeeded':
            wandb.termlog('Fine-tuning finished, logging metrics, model metadata, and more to W&B')
            return fine_tune
        if fine_tune.status == 'failed':
            wandb.termwarn(f'Fine-tune {fine_tune.id} has failed and will not be logged')
            return fine_tune
        if fine_tune.status == 'cancelled':
            wandb.termwarn(f'Fine-tune {fine_tune.id} was cancelled and will not be logged')
            return fine_tune
        time.sleep(10)
        fine_tune = cls.openai_client.fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tune.id)
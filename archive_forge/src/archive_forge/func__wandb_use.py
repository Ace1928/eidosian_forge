import inspect
import pickle
from functools import wraps
from pathlib import Path
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
@typedispatch
def _wandb_use(name: str, data: BaseEstimator, models=False, run=None, testing=False, *args, **kwargs):
    if testing:
        return 'models' if models else None
    if models:
        run.use_artifact(f'{name}:latest')
        wandb.termlog(f'Using artifact: {name} ({type(data)})')
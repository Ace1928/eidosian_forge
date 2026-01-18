import inspect
import pickle
from functools import wraps
from pathlib import Path
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
@typedispatch
def wandb_use(name: str, data: (dict, list, set, str, int, float, bool), *args, **kwargs):
    pass
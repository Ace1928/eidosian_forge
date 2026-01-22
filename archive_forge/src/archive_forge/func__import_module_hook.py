from typing import Optional, Sequence, Union
import wandb
from wandb.errors import UnsupportedError
from wandb.sdk import wandb_run
from wandb.sdk.lib.wburls import wburls
def _import_module_hook() -> None:
    """On wandb import, setup anything needed based on parent process require calls."""
    require('service')
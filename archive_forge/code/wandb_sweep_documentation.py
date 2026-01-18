import urllib.parse
from typing import Callable, Dict, Optional, Union
import wandb
from wandb import env
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps.utils import handle_sweep_config_violations
from . import wandb_login
Public sweep controller constructor.

    Usage:
        ```python
        import wandb

        tuner = wandb.controller(...)
        print(tuner.sweep_config)
        print(tuner.sweep_id)
        tuner.configure_search(...)
        tuner.configure_stopping(...)
        ```

    
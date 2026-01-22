import os
import re
import socket
from typing import Any, Optional
import wandb
import wandb.util
def _notify_tensorboard_logdir(logdir: str, save: bool=True, root_logdir: str='') -> None:
    if wandb.run is not None:
        wandb.run._tensorboard_callback(logdir, save=save, root_logdir=root_logdir)
import configparser
import getpass
import os
import tempfile
from typing import Any, Optional
from wandb import env
from wandb.old import core
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.runid import generate_id
def try_create_dir(path) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        if os.access(path, os.W_OK):
            return True
    except OSError:
        pass
    return False
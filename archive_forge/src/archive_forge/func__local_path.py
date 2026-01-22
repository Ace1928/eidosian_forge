import configparser
import getpass
import os
import tempfile
from typing import Any, Optional
from wandb import env
from wandb.old import core
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.runid import generate_id
@staticmethod
def _local_path(root_dir=None):
    filesystem.mkdir_exists_ok(core.wandb_dir(root_dir))
    return os.path.join(core.wandb_dir(root_dir), 'settings')
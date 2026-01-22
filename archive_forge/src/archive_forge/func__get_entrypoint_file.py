import enum
import logging
import os
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import wandb
import wandb.docker as docker
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch import utils
from wandb.sdk.lib.runid import generate_id
from .errors import LaunchError
from .utils import LOG_PREFIX, recursive_macro_sub
def _get_entrypoint_file(entrypoint: List[str]) -> Optional[str]:
    """Get the entrypoint file from the given command.

    Args:
        entrypoint (List[str]): List of command and arguments.

    Returns:
        Optional[str]: The entrypoint file if found, otherwise None.
    """
    if not entrypoint:
        return None
    if entrypoint[0].endswith('.py') or entrypoint[0].endswith('.sh'):
        return entrypoint[0]
    if len(entrypoint) < 2:
        return None
    return entrypoint[1]
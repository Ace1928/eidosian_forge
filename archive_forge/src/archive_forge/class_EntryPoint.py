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
class EntryPoint:
    """An entry point into a wandb launch specification."""

    def __init__(self, name: Optional[str], command: List[str]):
        self.name = name
        self.command = command

    def compute_command(self, user_parameters: Optional[List[str]]) -> List[str]:
        """Converts user parameter dictionary to a string."""
        ret = self.command
        if user_parameters:
            return ret + user_parameters
        return ret

    def update_entrypoint_path(self, new_path: str) -> None:
        """Updates the entrypoint path to a new path."""
        if len(self.command) == 2 and self.command[0] in ['python', 'bash']:
            self.command[1] = new_path
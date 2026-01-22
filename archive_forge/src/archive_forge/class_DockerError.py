import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
class DockerError(Error):
    """Raised when attempting to execute a docker command."""

    def __init__(self, command_launched: List[str], return_code: int, stdout: Optional[bytes]=None, stderr: Optional[bytes]=None) -> None:
        command_launched_str = ' '.join(command_launched)
        error_msg = f'The docker command executed was `{command_launched_str}`.\nIt returned with code {return_code}\n'
        if stdout is not None:
            error_msg += f"The content of stdout is '{stdout.decode()}'\n"
        else:
            error_msg += "The content of stdout can be found above the stacktrace (it wasn't captured).\n"
        if stderr is not None:
            error_msg += f"The content of stderr is '{stderr.decode()}'\n"
        else:
            error_msg += "The content of stderr can be found above the stacktrace (it wasn't captured)."
        super().__init__(error_msg)
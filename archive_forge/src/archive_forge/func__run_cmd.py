import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dockerpycreds.utils import find_executable  # type: ignore
import wandb
from wandb.apis.internal import Api
from wandb.sdk.lib import runid
from .._project_spec import LaunchProject
def _run_cmd(self, cmd: List[str], output_only: Optional[bool]=False) -> Optional[Union['subprocess.Popen[bytes]', bytes]]:
    """Run the command and returns a popen object or the stdout of the command.

        Arguments:
        cmd: The command to run
        output_only: If true just return the stdout bytes
        """
    try:
        env = os.environ
        popen = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE)
        if output_only:
            popen.wait()
            if popen.stdout is not None:
                return popen.stdout.read()
        return popen
    except subprocess.CalledProcessError as e:
        wandb.termerror(f'Command failed: {e}')
        return None
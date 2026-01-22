import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from getpass import getuser
from shlex import quote
from typing import Dict, List
import click
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.docker import (
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.subprocess_output_util import (
from ray.autoscaler.command_runner import CommandRunnerInterface
def _set_ssh_ip_if_required(self):
    if self.ssh_ip is not None:
        return
    deadline = time.time() + AUTOSCALER_NODE_START_WAIT_S
    with LogTimer(self.log_prefix + 'Got IP'):
        ip = self._wait_for_ip(deadline)
        cli_logger.doassert(ip is not None, 'Could not get node IP.')
        assert ip is not None, 'Unable to find IP of node'
    self.ssh_ip = ip
    try:
        os.makedirs(self.ssh_control_path, mode=448, exist_ok=True)
    except OSError as e:
        cli_logger.warning('{}', str(e))
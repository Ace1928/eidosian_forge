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
def _with_environment_variables(cmd: str, environment_variables: Dict[str, object]):
    """Prepend environment variables to a shell command.

    Args:
        cmd: The base command.
        environment_variables (Dict[str, object]): The set of environment
            variables. If an environment variable value is a dict, it will
            automatically be converted to a one line yaml string.
    """
    as_strings = []
    for key, val in environment_variables.items():
        val = json.dumps(val, separators=(',', ':'))
        s = 'export {}={};'.format(key, quote(val))
        as_strings.append(s)
    all_vars = ''.join(as_strings)
    return all_vars + cmd
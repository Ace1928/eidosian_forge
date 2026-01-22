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
class SSHOptions:

    def __init__(self, ssh_key, control_path=None, **kwargs):
        self.ssh_key = ssh_key
        self.arg_dict = {'StrictHostKeyChecking': 'no', 'UserKnownHostsFile': os.devnull, 'IdentitiesOnly': 'yes', 'ExitOnForwardFailure': 'yes', 'ServerAliveInterval': 5, 'ServerAliveCountMax': 3}
        if control_path:
            self.arg_dict.update({'ControlMaster': 'auto', 'ControlPath': '{}/%C'.format(control_path), 'ControlPersist': '10s'})
        self.arg_dict.update(kwargs)

    def to_ssh_options_list(self, *, timeout=60):
        self.arg_dict['ConnectTimeout'] = '{}s'.format(timeout)
        ssh_key_option = ['-i', self.ssh_key] if self.ssh_key else []
        return ssh_key_option + [x for y in (['-o', '{}={}'.format(k, v)] for k, v in self.arg_dict.items() if v is not None) for x in y]
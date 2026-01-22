import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
def _get_ssh_client(self):
    import paramiko
    config = paramiko.SSHConfig()
    config.parse(open(os.path.expanduser('~/.ssh/config')))
    host = config.lookup(self.inputs.hostname)
    if 'proxycommand' in host:
        proxy = paramiko.ProxyCommand(subprocess.check_output([os.environ['SHELL'], '-c', 'echo %s' % host['proxycommand']]).strip())
    else:
        proxy = None
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host['hostname'], username=host['user'], sock=proxy)
    return client
import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
def _get_base_ssh_command(self):
    cmd = ['ssh']
    if self.key_files:
        self.key_files = cast(str, self.key_files)
        cmd += ['-i', self.key_files]
    if self.timeout:
        cmd += ['-oConnectTimeout=%s' % self.timeout]
    cmd += ['{}@{}'.format(self.username, self.hostname)]
    return cmd
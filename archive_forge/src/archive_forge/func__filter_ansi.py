import os
import getpass
import json
import pty
import random
import re
import select
import string
import subprocess
import time
from functools import wraps
from ansible_collections.amazon.aws.plugins.module_utils.botocore import HAS_BOTO3
from ansible.errors import AnsibleConnectionFailure
from ansible.errors import AnsibleError
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves import xrange
from ansible.module_utils._text import to_bytes
from ansible.module_utils._text import to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import _common_args
from ansible.utils.display import Display
def _filter_ansi(self, line):
    """remove any ANSI terminal control codes"""
    line = to_text(line)
    if self.is_windows:
        osc_filter = re.compile('\\x1b\\][^\\x07]*\\x07')
        line = osc_filter.sub('', line)
        ansi_filter = re.compile('(\\x9B|\\x1B\\[)[0-?]*[ -/]*[@-~]')
        line = ansi_filter.sub('', line)
        line = line.replace('\r\r\n', '\n')
        if len(line) == 201:
            line = line[:-1]
    return line
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
def _flush_stderr(self, session_process):
    """read and return stderr with minimal blocking"""
    poll_stderr = select.poll()
    poll_stderr.register(session_process.stderr, select.POLLIN)
    stderr = ''
    while session_process.poll() is None:
        if not poll_stderr.poll(1):
            break
        line = session_process.stderr.readline()
        self._vvvv(f'stderr line: {to_text(line)}')
        stderr = stderr + line
    return stderr
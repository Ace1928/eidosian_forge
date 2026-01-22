from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import os.path
import subprocess
import threading
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
def _GetStdout(cmd, timeout_sec, show_stderr=True):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None if show_stderr else subprocess.PIPE)
    with _TimeoutThread(p.kill, timeout_sec):
        stdout, _ = p.communicate()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return six.ensure_text(stdout)
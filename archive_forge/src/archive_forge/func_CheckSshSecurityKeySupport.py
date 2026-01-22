from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import enum
import errno
import getpass
import os
import re
import string
import subprocess
import tempfile
import textwrap
from googlecloudsdk.api_lib.oslogin import client as oslogin_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.oslogin import oslogin_utils
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves.urllib.parse import quote
def CheckSshSecurityKeySupport():
    """Check the local SSH installation for security key support.

  Runs 'ssh -Q key' and looks for keys starting with 'sk-'.
  PuTTY on Windows will return False.

  Returns:
    True if SSH supports security keys, False if not, and None if support
    cannot be determined.
  """
    env = Environment.Current()
    if env.suite == Suite.PUTTY:
        return False
    ssh_flags = ['-Q', 'key']
    cmd = SSHCommand(None, extra_flags=ssh_flags, tty=False)
    cmd_list = cmd.Build()
    log.debug(cmd_list)
    try:
        output = six.ensure_str(subprocess.check_output(cmd_list, stderr=subprocess.STDOUT))
        log.debug(output)
    except subprocess.CalledProcessError:
        log.debug('Cannot determine SSH supported key types using command: {0}'.format(' '.join(cmd_list)))
        return None
    keys_supported = output.splitlines()
    log.debug('Supported SSH key types: {0}'.format(keys_supported))
    for key in keys_supported:
        if key.startswith('sk-'):
            return True
    return False
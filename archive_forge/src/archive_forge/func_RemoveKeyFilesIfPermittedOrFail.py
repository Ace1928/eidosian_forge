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
def RemoveKeyFilesIfPermittedOrFail(self, force_key_file_overwrite=None):
    """Removes all SSH key files if user permitted this behavior.

    Precondition: The SSH key files are currently in a broken state.

    Depending on `force_key_file_overwrite`, delete all SSH key files:

    - If True, delete key files.
    - If False, cancel immediately.
    - If None and
      - interactive, prompt the user.
      - non-interactive, cancel.

    Args:
      force_key_file_overwrite: bool or None, overwrite broken key files.

    Raises:
      console_io.OperationCancelledError: Operation intentionally cancelled.
      OSError: Error deleting the broken file(s).
    """
    message = 'Your SSH key files are broken.\n' + self._StatusMessage()
    if force_key_file_overwrite is False:
        raise console_io.OperationCancelledError(message + 'Operation aborted.')
    message += 'We are going to overwrite all above files.'
    log.warning(message)
    if force_key_file_overwrite is None:
        console_io.PromptContinue(default=False, cancel_on_no=True)
    for key_file in six.viewvalues(self.keys):
        try:
            os.remove(key_file.filename)
        except OSError as e:
            if e.errno == errno.EISDIR:
                raise
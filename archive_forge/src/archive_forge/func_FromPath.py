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
@classmethod
def FromPath(cls, path):
    """Convert an SCP-style positional argument to a file reference.

    Note that this method does not raise. No lookup of either local or remote
    file presence exists.

    Args:
      path: str, A path on the canonical scp form `[remote:]path`. If remote,
        `path` can be empty, e.g. `me@host:`.

    Returns:
      FileReference, the constructed object.
    """
    local_drive = os.path.splitdrive(path)[0]
    remote_arg, sep, file_path = path.partition(':')
    remote = Remote.FromArg(remote_arg) if sep else None
    if remote and (not local_drive):
        return cls(path=file_path, remote=remote)
    else:
        return cls(path=path)
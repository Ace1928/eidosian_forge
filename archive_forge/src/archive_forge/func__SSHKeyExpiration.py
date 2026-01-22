from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
def _SSHKeyExpiration(ssh_key):
    """Returns a datetime expiration time for an ssh key entry from metadata.

  Args:
    ssh_key: A single ssh key entry.

  Returns:
    None if no expiration set or a datetime object of the expiration (in UTC).

  Raises:
    ValueError: If the ssh key entry could not be parsed for expiration (invalid
      format, missing expected entries, etc).
    dateutil.DateTimeSyntaxError: The found expiration could not be parsed.
    dateutil.DateTimeValueError: The found expiration could not be parsed.
  """
    key_parts = ssh_key.split()
    if len(key_parts) < 4 or key_parts[2] != 'google-ssh':
        return None
    expiration_json = ' '.join(key_parts[3:])
    expiration = json.loads(expiration_json)
    try:
        expireon = times.ParseDateTime(expiration['expireOn'])
    except KeyError:
        raise ValueError('Unable to find expireOn entry')
    return times.LocalizeDateTime(expireon, times.UTC)
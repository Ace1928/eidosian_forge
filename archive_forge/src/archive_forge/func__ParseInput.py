from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import sys
import textwrap
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from oauth2client import client
import six
def _ParseInput(self):
    """Parse the fields from stdin.

    Returns:
      {str: str}, The parsed parameters given on stdin.
    """
    info = {}
    for line in sys.stdin:
        if _BLANK_LINE_RE.match(line):
            continue
        match = _KEYVAL_RE.match(line)
        if not match:
            raise auth_exceptions.GitCredentialHelperError('Invalid input line format: [{format}].'.format(format=line.rstrip('\n')))
        key, val = match.groups()
        info[key] = val.strip()
    if 'protocol' not in info:
        raise auth_exceptions.GitCredentialHelperError('Required key "protocol" missing.')
    if 'host' not in info:
        raise auth_exceptions.GitCredentialHelperError('Required key "host" missing.')
    if info.get('protocol') != 'https':
        raise auth_exceptions.GitCredentialHelperError('Invalid protocol [{p}].  "https" expected.'.format(p=info.get('protocol')))
    return info
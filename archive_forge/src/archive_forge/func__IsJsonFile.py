from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.auth import service_account as auth_service_account
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _IsJsonFile(filename):
    """Check and validate if given filename is proper json file."""
    content = console_io.ReadFromFileOrStdin(filename, binary=True)
    try:
        return (json.loads(encoding.Decode(content)), True)
    except ValueError as e:
        if filename.endswith('.json'):
            raise auth_service_account.BadCredentialFileException('Could not read json file {0}: {1}'.format(filename, e))
    return (content, False)
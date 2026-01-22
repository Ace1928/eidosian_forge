from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import json
import time
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.diagnose import diagnose_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def _ConstructDiagnosticsKeyEntry(self, signed_url, trace):
    """Generates a JSON String that is a command for the VM to extract the logs.

    Args:
      signed_url: The url where the logs can be uploaded.
      trace: Whether or not to take a 10 minute trace on the VM.
    Returns:
      A JSON String that can be written to the metadata server to trigger the
      extraction of logs.
    """
    expire_str = time_util.CalculateExpiration(300)
    diagnostics_key_data = {'signedUrl': signed_url, 'trace': trace, 'expireOn': expire_str}
    return json.dumps(diagnostics_key_data, sort_keys=True)
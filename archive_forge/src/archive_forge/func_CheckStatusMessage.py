from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Optional
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations import integration_printer
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def CheckStatusMessage(release_track, integration_name):
    """Message about check status with describe command.

  Args:
    release_track: Release track of the command being run.
    integration_name: str, name of the integration.

  Returns:
    A formatted string of the message.
  """
    return 'You can check the status with `gcloud {}run integrations describe {}`'.format(_ReleaseCommandPrefix(release_track), integration_name)
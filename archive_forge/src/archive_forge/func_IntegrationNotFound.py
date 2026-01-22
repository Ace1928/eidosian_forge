from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Optional
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations import integration_printer
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def IntegrationNotFound(name):
    """Generates a message when an integration is not found.

  Args:
    name: name of the integration.

  Returns:
    A string message.
  """
    return 'Integration [{}] cannot be found. First create an integration with `gcloud run integrations create`.'.format(name)
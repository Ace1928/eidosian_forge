from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.models import client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
Runs command with model client.

    Concrete gCloud SDK command subclasses are required to override this.

    Args:
      args: Command arguments.
      model_ref: The model resource reference.
      region: The region of the model resource reference.

    Returns:
      The response from running the given command with model client.
    
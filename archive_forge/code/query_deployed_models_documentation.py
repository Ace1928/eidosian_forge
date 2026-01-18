from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.deployment_resource_pools import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
Queries Vertex AI deployed models sharing a specified deployment resource pool.

  This command queries deployed models sharing the specified deployment resource
  pool.

  ## EXAMPLES

  To query a deployment resource pool with name ''example'' in region
  ''us-central1'', run:

    $ {command} example --region=us-central1
  
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.endpoints import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import endpoints_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.ai import region_util
Undeploy a model from an existing Vertex AI endpoint.

  ## EXAMPLES

  To undeploy a model ``456'' from an endpoint ``123'' under project ``example''
  in region ``us-central1'', run:

    $ {command} 123 --project=example --region=us-central1
    --deployed-model-id=456
  
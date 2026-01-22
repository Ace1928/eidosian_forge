from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.endpoints import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import endpoints_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ExplainBeta(ExplainGa):
    """Request an online explanation from an Vertex AI endpoint.

     `{command}` sends an explanation request to the Vertex AI endpoint for
     the given instances. This command reads up to 100 instances, though the
     service itself accepts instances up to the payload limit size
     (currently, 1.5MB).

     ## EXAMPLES

     To send an explanation request to the endpoint for the json file,
     input.json, run:

     $ {command} ENDPOINT_ID --region=us-central1 --json-request=input.json
  """

    def Run(self, args):
        return _Run(args, constants.BETA_VERSION)
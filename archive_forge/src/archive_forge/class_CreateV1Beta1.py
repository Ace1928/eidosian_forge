from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.deployment_resource_pools import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import deployment_resource_pools_util
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core import log
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class CreateV1Beta1(base.CreateCommand):
    """Create a new Vertex AI deployment resource pool.

  This command creates a new deployment resource pool with a provided deployment
  resource pool id (name) in a provided region (assuming that region provides
  support for this api). You can choose to simply provide the resource pool
  name and create an instance with default arguments, or you can pass in your
  own preferences, such as the accelerator and machine specs. Please note this
  functionality is not yet available in the GA track and is currently only
  part of the v1beta1 API version.

  ## EXAMPLES

  To create a deployment resource pool with name ``example'' in region
  ``us-central1'', run:

    $ {command} example --region=us-central1
  """

    @staticmethod
    def Args(parser):
        return _AddArgsBeta(parser)

    def Run(self, args):
        return _RunBeta(args)
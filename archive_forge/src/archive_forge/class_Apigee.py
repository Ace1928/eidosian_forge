from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Apigee(base.Group):
    """Manage Apigee resources.

  Commands for managing Google Cloud Apigee resources.
  """
    category = base.API_PLATFORM_AND_ECOSYSTEMS_CATEGORY
    detailed_help = {'DESCRIPTION': 'Manage Apigee resources.', 'EXAMPLES': "\n          To list API proxies in the active Cloud Platform project, run:\n\n            $ {command} apis list\n\n          To deploy an API proxy named ``hello-world'' to the ``test''\n          environment, run:\n\n            $ {command} apis deploy --environment=test --api=hello-world\n\n          To get the status of that deployment, run:\n\n            $ {command} deployments describe --environment=test --api=hello-world\n\n          To undeploy that API proxy, run:\n\n            $ {command} apis undeploy --environment=test --api=hello-world\n          "}

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
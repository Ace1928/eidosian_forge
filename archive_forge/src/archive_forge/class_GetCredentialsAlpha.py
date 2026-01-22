from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class GetCredentialsAlpha(base.Command):
    """Fetch credentials for a running Anthos Config Controller.

  {command} updates a `kubeconfig` file with appropriate credentials and
  endpoint information to point `kubectl` at a specific
  Anthos Config Controller.
  """
    detailed_help = {'EXAMPLES': "          To switch to working on your Config Controller 'main', run:\n\n            $ {command} main --location=us-central1\n      "}

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        parser.add_argument('name', help='Name of the Anthos Config Controller.')
        parser.add_argument('--location', required=True, help='The location (region) of the Anthos Config Controller.')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      container_util.Error: if the cluster is unreachable or not running.
    """
        cluster, cluster_ref = _BaseRun(args)
        container_util.ClusterConfig.Persist(cluster, cluster_ref.projectId)
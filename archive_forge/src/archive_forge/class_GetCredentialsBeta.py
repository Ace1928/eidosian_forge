from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class GetCredentialsBeta(base.Command):
    """Fetch credentials for a running cluster.

  {command} updates a `kubeconfig` file with appropriate credentials and
  endpoint information to point `kubectl` at a specific cluster in Google
  Kubernetes Engine.

  It takes a project and a zone as parameters, passed through by set
  defaults or flags. By default, credentials are written to `HOME/.kube/config`.
  You can provide an alternate path by setting the `KUBECONFIG` environment
  variable. If `KUBECONFIG` contains multiple paths, the first one is used.

  This command enables switching to a specific cluster, when working
  with multiple clusters. It can also be used to access a previously created
  cluster from a new workstation.

  By default, {command} will configure kubectl to automatically refresh its
  credentials using the same identity as gcloud. If you are running kubectl as
  part of an application, it is recommended to use [application default
  credentials](https://cloud.google.com/docs/authentication/production).
  To configure a `kubeconfig` file to use application default credentials, set
  the container/use_application_default_credentials
  [Cloud SDK property](https://cloud.google.com/sdk/docs/properties) to true
  before running {command}

  See [](https://cloud.google.com/kubernetes-engine/docs/kubectl) for
  kubectl documentation.
  """
    detailed_help = {'EXAMPLES': "          To switch to working on your cluster 'sample-cluster', run:\n\n            $ {command} sample-cluster --location=us-central1-f\n      "}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        flags.AddGetCredentialsArgs(parser)
        flags.AddCrossConnectSubnetworkFlag(parser)
        flags.AddPrivateEndpointFQDNFlag(parser)
        flags.AddDnsEndpointFlag(parser)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      util.Error: if the cluster is unreachable or not running.
    """
        flags.VerifyGetCredentialsFlags(args)
        cluster, cluster_ref = _BaseRun(args, self.context)
        util.ClusterConfig.Persist(cluster, cluster_ref.projectId, args.internal_ip, args.cross_connect_subnetwork, args.private_endpoint_fqdn, args.dns_endpoint)
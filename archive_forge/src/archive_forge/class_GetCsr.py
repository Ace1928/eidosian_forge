from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.privateca import resource_args
@base.ReleaseTracks(base.ReleaseTrack.GA)
class GetCsr(base.Command):
    """Get the CSR for a subordinate certificate authority that has not yet been activated.

  Gets the PEM-encoded CSR for a subordinate certificate authority that is
  awaiting user activation. The CSR should be signed by the issuing Certificate
  Authority and uploaded back using the `subordinates activate` command.

  ## EXAMPLES

    To download the CSR for the 'server-tls-1' CA into a file called
    'server-tls-1.csr':

      $ {command} server-tls-1 --location=us-west1 --pool=my-pool > server-tls-1.csr
  """

    @staticmethod
    def Args(parser):
        resource_args.AddCertAuthorityPositionalResourceArg(parser, 'for which to get the CSR')
        parser.display_info.AddFormat('value(pemCsr)')

    def Run(self, args):
        client = privateca_base.GetClientInstance(api_version='v1')
        messages = privateca_base.GetMessagesModule(api_version='v1')
        ca_ref = args.CONCEPTS.certificate_authority.Parse()
        return client.projects_locations_caPools_certificateAuthorities.Fetch(messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesFetchRequest(name=ca_ref.RelativeName()))
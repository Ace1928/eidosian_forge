from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.auth import enterprise_certificate_config
from googlecloudsdk.command_lib.auth.flags import AddCommonEnterpriseCertConfigFlags
class CreateWindowsConfig(base.CreateCommand):
    """Create an enterprise-certificate configuration file for Windows.

  This command creates a configuration file used by gcloud to use the
  enterprise-certificate-proxy component for mTLS.
  """
    detailed_help = {'EXAMPLES': textwrap.dedent('          To create a credential configuration run:\n\n            $ {command} --issuer=$CERT_ISSUER --store=$STORE --provider=$PROVIDER\n          ')}

    @classmethod
    def Args(cls, parser):
        AddCommonEnterpriseCertConfigFlags(parser)
        parser.add_argument('--issuer', help='The certificate issuer.', required=True)
        parser.add_argument('--store', help='The Windows secure store.', required=True)
        parser.add_argument('--provider', help='The Windows secure store provider.', required=True)

    def Run(self, args):
        enterprise_certificate_config.create_config(enterprise_certificate_config.ConfigType.MYSTORE, **vars(args))
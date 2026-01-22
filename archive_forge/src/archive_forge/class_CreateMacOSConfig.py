from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.auth import enterprise_certificate_config
from googlecloudsdk.command_lib.auth.flags import AddCommonEnterpriseCertConfigFlags
class CreateMacOSConfig(base.CreateCommand):
    """Create an enterprise-certificate configuration file for MacOS.

  This command creates a configuration file used by gcloud to use the
  enterprise-certificate-proxy component for mTLS.
  """
    detailed_help = {'EXAMPLES': textwrap.dedent('          To create a credential configuration run:\n\n            $ {command} --issuer=$CERT_ISSUER\n          ')}

    @classmethod
    def Args(cls, parser):
        AddCommonEnterpriseCertConfigFlags(parser)
        parser.add_argument('--issuer', help='The certificate issuer.', required=True)

    def Run(self, args):
        enterprise_certificate_config.create_config(enterprise_certificate_config.ConfigType.KEYCHAIN, **vars(args))
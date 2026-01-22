from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetIdpCertificateAuthorityDataFlag():
    """Anthos auth token idp-certificate-authority-data flag, specifies the PEM-encoded certificate authority certificate for OIDC provider."""
    return base.Argument('--idp-certificate-authority-data', required=False, help='PEM-encoded certificate authority certificate for OIDC provider.')
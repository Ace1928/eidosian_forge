from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetIdpIssuerUrlFlag():
    """Anthos auth token idp-issuer-url flag, specifies the URI for the OIDC provider."""
    return base.Argument('--idp-issuer-url', required=False, help='URI for the OIDC provider. This URI should point to the level below .well-known/openid-configuration.')
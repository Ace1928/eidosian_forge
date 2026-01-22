from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddOidcConfig(parser):
    """Adds Oidc Config flags.

  Args:
    parser: The argparse.parser to add the arguments to.
  """
    group = parser.add_group('OIDC config', required=True)
    AddIssuerUrl(group, required=True)
    AddOidcJwks(group)
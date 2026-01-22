from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddHasPrivateIssuer(parser):
    help_text = "Indicates no publicly routable OIDC discovery endpoint exists\nfor the Kubernetes service account token issuer.\n\nIf this flag is set, gcloud will read the issuer URL and JWKs from the cluster's\napi server.\n"
    parser.add_argument('--has-private-issuer', help=help_text, action='store_true')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetSslPolicyArgumentForOtherResource(proxy_type, required=False):
    """Returns the flag for specifying the SSL policy."""
    return compute_flags.ResourceArgument(name='--ssl-policy', resource_name='SSL policy', completer=SslPoliciesCompleter, plural=False, required=required, global_collection='compute.sslPolicies', short_help='A reference to an SSL policy resource that defines the server-side support for SSL features.', detailed_help='        A reference to an SSL policy resource that defines the server-side\n        support for SSL features and affects the connections between clients\n        and load balancers that are using the {proxy_type} proxy. The SSL\n        policy must exist and cannot be\n        deleted while referenced by a target {proxy_type} proxy.\n        '.format(proxy_type=proxy_type))
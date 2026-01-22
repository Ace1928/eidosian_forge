from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import scope as compute_scope
def ResolveSslPolicy(args, ssl_policy_arg, target_ssl_proxy_ref, resources):
    """Parses the SSL policies that are pointed to by a Target SSL Proxy from args.

  This function handles parsing regional/global SSL policies that are
  pointed to by a regional/global Target SSL Proxy.

  Args:
    args: The arguments provided to the target_ssl_proxies command.
    ssl_policy_arg: The ResourceArgument specification for the ssl_policies
      argument.
    target_ssl_proxy_ref: The resource reference to the Target SSL Proxy. This
      is obtained by parsing the Target SSL Proxy arguments provided.
    resources: ComputeApiHolder resources.

  Returns:
    Returns the SSL policy resource
  """
    if not compute_scope.IsSpecifiedForFlag(args, 'ssl_policy'):
        if IsRegionalTargetSslProxiesRef(target_ssl_proxy_ref):
            args.ssl_policy_region = target_ssl_proxy_ref.region
        else:
            args.global_ssl_policy = bool(args.ssl_policy)
    return ssl_policy_arg.ResolveAsResource(args, resources)
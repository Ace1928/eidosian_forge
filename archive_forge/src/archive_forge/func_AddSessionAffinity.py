from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSessionAffinity(parser, target_pools=False, hidden=False, support_client_only=False):
    """Adds session affinity flag to the argparse.

  Args:
    parser: An argparse.ArgumentParser instance.
    target_pools: Indicates if the backend pool is target pool.
    hidden: if hidden=True, retains help but does not display it.
    support_client_only: Indicates if CLIENT_IP_NO_DESTINATION is valid choice.
  """
    choices = {'CLIENT_IP': "Route requests to instances based on the hash of the client's IP address.", 'NONE': 'Session affinity is disabled.', 'CLIENT_IP_PROTO': 'Connections from the same client IP with the same IP protocol will go to the same VM in the pool while that VM remains healthy.'}
    if not target_pools:
        choices.update({'GENERATED_COOKIE': '(Applicable if `--load-balancing-scheme` is `INTERNAL_MANAGED`, `INTERNAL_SELF_MANAGED`, `EXTERNAL_MANAGED`, or `EXTERNAL`)  If the `--load-balancing-scheme` is `EXTERNAL` or `EXTERNAL_MANAGED`, routes requests to backend VMs or endpoints  in a NEG, based on the contents of the `GCLB` cookie set by the  load balancer. Only applicable when `--protocol` is HTTP, HTTPS,  or HTTP2. If the `--load-balancing-scheme` is `INTERNAL_MANAGED`  or `INTERNAL_SELF_MANAGED`, routes requests to backend VMs or  endpoints in a NEG, based on the contents of the `GCILB` cookie  set by the proxy. (If no cookie is present, the proxy  chooses a backend VM or endpoint and sends a `Set-Cookie`  response for future requests.) If the `--load-balancing-scheme`  is `INTERNAL_SELF_MANAGED`, routes requests to backend VMs or  endpoints in a NEG, based on the contents of a cookie set by  Traffic Director. This session affinity is only valid if the  load balancing locality policy is either `RING_HASH` or  `MAGLEV`.', 'CLIENT_IP_PROTO': '(Applicable if `--load-balancing-scheme` is `INTERNAL`) Connections from the same client IP with the same IP protocol will go to the same backend VM while that VM remains healthy.', 'CLIENT_IP_PORT_PROTO': '(Applicable if `--load-balancing-scheme` is `INTERNAL`) Connections from the same client IP with the same IP protocol and port will go to the same backend VM while that VM remains healthy.', 'HTTP_COOKIE': "(Applicable if `--load-balancing-scheme` is `INTERNAL_MANAGED`, `EXTERNAL_MANAGED` or `INTERNAL_SELF_MANAGED`) Route requests to  backend VMs or  endpoints in a NEG, based on an HTTP cookie named in the  `HTTP_COOKIE` flag (with the optional `--affinity-cookie-ttl`  flag). If the client has not provided the cookie,  the proxy generates the cookie and returns it to the client in a  `Set-Cookie` header. This session affinity is only valid if the  load balancing locality policy is either `RING_HASH` or `MAGLEV`  and the backend service's consistent hash specifies the HTTP  cookie.", 'HEADER_FIELD': "(Applicable if `--load-balancing-scheme` is `INTERNAL_MANAGED`, `EXTERNAL_MANAGED`, or `INTERNAL_SELF_MANAGED`) Route requests  to backend VMs or  endpoints in a NEG based on the value of the HTTP header named  in the `--custom-request-header` flag. This session  affinity is only valid if the load balancing locality policy  is either `RING_HASH` or `MAGLEV` and the backend service's  consistent hash specifies the name of the HTTP header."})
        if support_client_only:
            choices.update({'CLIENT_IP_NO_DESTINATION': "Directs a particular client's request to the same backend VM based on a hash created on the client's IP address only. This is used in L4 ILB as Next-Hop scenarios. It differs from the Client-IP option in that Client-IP uses a hash based on both client-IP's address and destination address."})
    help_str = 'The type of session affinity to use. Supports both TCP and UDP.'
    parser.add_argument('--session-affinity', choices=choices, default='NONE' if target_pools else None, type=lambda x: x.upper(), hidden=hidden, help=help_str)
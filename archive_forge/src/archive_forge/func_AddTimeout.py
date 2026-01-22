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
def AddTimeout(parser, default='30s'):
    parser.add_argument('--timeout', default=default, type=arg_parsers.Duration(), help="      Applicable to all load balancing products except passthrough Network Load\n      Balancers. For internal passthrough Network Load Balancers\n      (``load-balancing-scheme'' set to INTERNAL) and\n      external passthrough Network Load Balancers (``global'' not set and\n      ``load-balancing-scheme'' set to EXTERNAL), ``timeout'' is ignored.\n\n      If the ``protocol'' is HTTP, HTTPS, or HTTP2, ``timeout'' is a\n      request/response timeout for HTTP(S) traffic, meaning the amount\n      of time that the load balancer waits for a backend to return a\n      full response to a request. If WebSockets traffic is supported, the\n      ``timeout'' parameter sets the maximum amount of time that a\n      WebSocket can be open (idle or not).\n\n      For example, for HTTP, HTTPS, or HTTP2 traffic, specifying a ``timeout''\n      of 10s means that backends have 10 seconds to respond to the load\n      balancer's requests. The load balancer retries the HTTP GET request one\n      time if the backend closes the connection or times out before sending\n      response headers to the load balancer. If the backend sends response\n      headers or if the request sent to the backend is not an HTTP GET request,\n      the load balancer does not retry. If the backend does not reply at all,\n      the load balancer returns a 502 Bad Gateway error to the client.\n\n      If the ``protocol'' is SSL or TCP, ``timeout'' is an idle timeout.\n\n      The full range of timeout values allowed is 1 - 2,147,483,647 seconds.\n      ")
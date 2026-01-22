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
def AddEnableCdn(parser):
    parser.add_argument('--enable-cdn', action=arg_parsers.StoreTrueFalseAction, help="      Enable or disable Cloud CDN for the backend service. Only available for\n      backend services with --load-balancing-scheme=EXTERNAL that use a\n      --protocol of HTTP, HTTPS, or HTTP2. Cloud CDN caches HTTP responses at\n      the edge of Google's network. Cloud CDN is disabled by default.\n      ")
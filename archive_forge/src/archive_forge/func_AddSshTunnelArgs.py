from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.core import log
def AddSshTunnelArgs(parser):
    parser.add_argument('--tunnel-through-iap', action='store_true', help='      Tunnel the ssh connection through Identity-Aware Proxy for TCP forwarding.\n\n      To learn more, see the\n      [IAP for TCP forwarding documentation](https://cloud.google.com/iap/docs/tcp-forwarding-overview).\n      ')
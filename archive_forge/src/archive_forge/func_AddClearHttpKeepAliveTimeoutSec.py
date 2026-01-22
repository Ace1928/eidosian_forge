from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddClearHttpKeepAliveTimeoutSec(parser):
    """Adds the http keep alive timeout sec argument."""
    parser.add_argument('--clear-http-keep-alive-timeout-sec', action='store_true', default=False, help='      Clears the previously configured HTTP keepalive timeout.\n      ')
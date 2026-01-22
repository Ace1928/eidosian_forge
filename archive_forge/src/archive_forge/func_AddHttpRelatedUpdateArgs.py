from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddHttpRelatedUpdateArgs(parser, include_weighted_load_balancing=False):
    """Adds parser arguments for update subcommands related to HTTP."""
    _AddPortRelatedUpdateArgs(parser)
    AddProxyHeaderRelatedUpdateArgs(parser)
    parser.add_argument('--host', help='      The value of the host header used in this HTTP health check request.\n      The host header is empty by default. When empty, the health check will set\n      the host header to the IP address of the backend VM or endpoint. You can\n      set the host header to an empty value to return to this default behavior.\n      ')
    parser.add_argument('--request-path', help="      The request path that this health check monitors. For example,\n      ``/healthcheck''.\n      ")
    if include_weighted_load_balancing:
        parser.add_argument('--weight-report-mode', choices=['ENABLE', 'DISABLE', 'DRY_RUN'], help='        Defines whether Weighted Load Balancing is enabled.\n        ')
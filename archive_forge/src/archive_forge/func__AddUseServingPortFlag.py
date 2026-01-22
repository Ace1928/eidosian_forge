from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def _AddUseServingPortFlag(parser, use_port_name=True):
    """Adds parser argument for using serving port option."""
    parser.add_argument('--use-serving-port', action='store_true', help='      If given, use the "serving port" for health checks:\n\n        - When health checking network endpoints in a Network Endpoint\n          Group, use the port specified with each endpoint.\n          `--use-serving-port` must be used when using a Network Endpoint Group\n          as a backend as this flag specifies the `portSpecification` option for\n          a Health Check object.\n        - When health checking other backends, use the port%s of\n          the backend service.' % (' or named port' if use_port_name else ''))
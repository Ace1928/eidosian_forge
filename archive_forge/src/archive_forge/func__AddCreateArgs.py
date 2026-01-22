from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import constants
from googlecloudsdk.command_lib.ml_engine import endpoint_util
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _AddCreateArgs(parser, support_console_logging=False):
    """Get arguments for the `ai-platform models create` command."""
    flags.GetModelName().AddToParser(parser)
    flags.GetDescriptionFlag('model').AddToParser(parser)
    region_group = parser.add_mutually_exclusive_group()
    region_group.add_argument('--region', choices=constants.SUPPORTED_REGIONS_WITH_GLOBAL, help=_REGION_FLAG_HELPTEXT)
    region_group.add_argument('--regions', metavar='REGION', type=arg_parsers.ArgList(min_length=1), help="The Google Cloud region where the model will be deployed (currently only a\nsingle region is supported) against the global endpoint.\n\nIf you specify this flag, do not specify `--region`.\n\nDefaults to 'us-central1' while using the global endpoint.\n")
    parser.add_argument('--enable-logging', action='store_true', help='If set, enables StackDriver Logging for online prediction. These logs are like standard server access logs, containing information such as timestamps and latency for each request.')
    if support_console_logging:
        parser.add_argument('--enable-console-logging', action='store_true', help='If set, enables StackDriver Logging of stderr and stdout streams for online prediction. These logs are more verbose than the standard access logs and can be helpful for debugging.')
    labels_util.AddCreateLabelsFlags(parser)
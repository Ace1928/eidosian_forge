from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.deployment_manager import dm_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPropertiesFlag(parser):
    """Add properties flag."""
    parser.add_argument('--properties', help='A comma separated, key:value, map to be used when deploying a template file or composite type directly.', type=arg_parsers.ArgDict(operators=dm_api_util.NewParserDict()), dest='properties')
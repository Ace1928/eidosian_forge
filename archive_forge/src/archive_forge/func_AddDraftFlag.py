from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDraftFlag(parser, hidden=False):
    """Add --draft flag."""
    parser.add_argument('--draft', hidden=hidden, help='If this flag is set to true, the exported deployment state file will be the draft state', action='store_true')
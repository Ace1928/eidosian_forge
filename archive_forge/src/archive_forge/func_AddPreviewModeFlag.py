from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddPreviewModeFlag(parser, hidden=False):
    """Add --preview-mode flag."""
    parser.add_argument('--preview-mode', hidden=hidden, help='Preview mode to set it to either default or delete.')
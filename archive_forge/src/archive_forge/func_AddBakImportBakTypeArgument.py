from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddBakImportBakTypeArgument(parser):
    """Add the 'bak-type' argument to the parser for bak import."""
    choices = [messages.ImportContext.BakImportOptionsValue.BakTypeValueValuesEnum.FULL.name, messages.ImportContext.BakImportOptionsValue.BakTypeValueValuesEnum.DIFF.name, messages.ImportContext.BakImportOptionsValue.BakTypeValueValuesEnum.TLOG.name]
    help_text = 'Type of bak file that will be imported. Applicable to SQL Server only.'
    parser.add_argument('--bak-type', choices=choices, required=False, default=messages.ImportContext.BakImportOptionsValue.BakTypeValueValuesEnum.FULL.name, help=help_text)
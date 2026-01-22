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
def AddReplication(parser, hidden=False):
    base.ChoiceArgument('--replication', required=False, choices=['synchronous', 'asynchronous'], default=None, help_str='Type of replication this instance uses. The default is synchronous.', hidden=hidden).AddToParser(parser)
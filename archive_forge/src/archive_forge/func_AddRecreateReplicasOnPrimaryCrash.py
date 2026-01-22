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
def AddRecreateReplicasOnPrimaryCrash(parser, hidden=False):
    """Adds --recreate-replicas-on-primary-crash flag."""
    parser.add_argument('--recreate-replicas-on-primary-crash', required=False, help='Allow/Disallow replica recreation when a primary MySQL instance operating in reduced durability mode crashes. Not recreating the replicas might lead to data inconsistencies between the primary and its replicas. This setting is only applicable for MySQL instances and is enabled by default.', action=arg_parsers.StoreTrueFalseAction, hidden=hidden)
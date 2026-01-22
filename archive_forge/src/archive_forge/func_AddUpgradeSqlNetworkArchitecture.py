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
def AddUpgradeSqlNetworkArchitecture(parser):
    """Adds --upgrade-sql-network-architecture flag."""
    kwargs = _GetKwargsForBoolFlag(False)
    parser.add_argument('--upgrade-sql-network-architecture', required=False, help='Upgrade from old network architecture to new network architecture. The\n       new network architecture offers better isolation, reliability, and faster\n       new feature adoption.', **kwargs)
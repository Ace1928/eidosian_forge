from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib import tasks as tasks_api_lib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMaxTasksToLeaseFlag(parser):
    base.Argument('--limit', type=int, default=1000, category=base.LIST_COMMAND_FLAGS, help='      The maximum number of tasks to lease. The maximum that can be requested is\n      1000.\n      ').AddToParser(parser)
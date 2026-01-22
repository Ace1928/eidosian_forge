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
def AddFilterLeasedTasksFlag(parser):
    tag_filter_group = parser.add_mutually_exclusive_group()
    tag_filter_group.add_argument('--tag', help='      A tag to filter each task to be leased. If a task has the tag and the\n      task is available to be leased, then it is listed and leased.\n      ')
    tag_filter_group.add_argument('--oldest-tag', action='store_true', help='      Only lease tasks which have the same tag as the task with the oldest\n      schedule time.\n      ')
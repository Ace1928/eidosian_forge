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
def _AlphaPushQueueFlags():
    return _BasePushQueueFlags() + [base.Argument('--max-tasks-dispatched-per-second', type=float, help='          The maximum rate at which tasks are dispatched from this queue.\n          '), base.Argument('--max-concurrent-tasks', type=int, help='          The maximum number of concurrent tasks that Cloud Tasks allows to\n          be dispatched for this queue. After this threshold has been reached,\n          Cloud Tasks stops dispatching tasks until the number of outstanding\n          requests decreases.\n          ')]
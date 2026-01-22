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
def _HttpPushQueueFlags():
    return [base.Argument('--http-uri-override', type=arg_parsers.ArgDict(key_type=_GetHttpUriOverrideKeysValidator(), min_length=1, max_length=6, operators={':': None}), metavar='KEY:VALUE', help='          If provided, the specified HTTP target URI override is used for all\n          tasks in the queue depending on what is set as the mode.\n          Allowed values for mode are: ALWAYS, IF_NOT_EXISTS. If not set, mode\n          defaults to ALWAYS.\n\n          KEY must be at least one of: [{}]. Any missing keys will use the\n          default.\n          '.format(', '.join(constants.HTTP_URI_OVERIDE_KEYS))), base.Argument('--http-method-override', help='          If provided, the specified HTTP method type override is used for\n          all tasks in the queue, no matter what is set at the task-level.\n          '), base.Argument('--http-header-override', metavar='HEADER_FIELD: HEADER_VALUE', action='append', type=_GetHeaderArgValidator(), help='          If provided, the specified HTTP headers override the existing\n          headers for all tasks in the queue.\n          If a task has a header with the same Key as a queue-level header\n          override, then the value of the task header will be overriden with\n          the value of the queue-level header. Otherwise, the queue-level\n          header will be added to the task headers.\n          Header values can contain commas. This flag can be repeated.\n          Repeated header fields will have their values overridden.\n          ')]
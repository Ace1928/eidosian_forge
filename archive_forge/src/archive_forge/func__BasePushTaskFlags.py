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
def _BasePushTaskFlags():
    return _CommonTaskFlags() + [base.Argument('--method', help='          The HTTP method to use for the request. If not specified, "POST" will\n          be used.\n          '), base.Argument('--header', metavar='HEADER_FIELD: HEADER_VALUE', action='append', type=_GetHeaderArgValidator(), help='          An HTTP request header. Header values can contain commas. This flag\n          can be repeated. Repeated header fields will have their values\n          overridden.\n          ')]
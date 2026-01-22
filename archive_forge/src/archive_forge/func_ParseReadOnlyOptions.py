from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import database_sessions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner import resource_args
from googlecloudsdk.command_lib.spanner import sql
from googlecloudsdk.command_lib.spanner.sql import QueryHasDml
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseReadOnlyOptions(self, args):
    """Parses the options for a read-only request from command line arguments.

    Args:
      args: Command line arguments.

    Returns:
      A ReadOnly message if the query is read-only (not DML), otherwise None.
    """
    if QueryHasDml(args.sql):
        if args.IsSpecified('strong'):
            raise c_exceptions.InvalidArgumentException('--strong', 'A timestamp bound cannot be specified for a DML statement.')
        if args.IsSpecified('read_timestamp'):
            raise c_exceptions.InvalidArgumentException('--read-timestamp', 'A timestamp bound cannot be specified for a DML statement.')
        return None
    else:
        msgs = apis.GetMessagesModule('spanner', 'v1')
        if args.IsSpecified('read_timestamp'):
            return msgs.ReadOnly(readTimestamp=args.read_timestamp)
        elif args.IsSpecified('strong'):
            if not args.strong:
                raise c_exceptions.InvalidArgumentException('--strong', '`--strong` cannot be set to false. Instead specify a different type of timestamp bound.')
            else:
                return msgs.ReadOnly(strong=True)
        else:
            return msgs.ReadOnly(strong=True)
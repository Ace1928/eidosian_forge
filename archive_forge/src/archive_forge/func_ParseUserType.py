from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
def ParseUserType(sql_messages, args):
    if args.type:
        return sql_messages.User.TypeValueValuesEnum.lookup_by_name(args.type.upper())
    return None
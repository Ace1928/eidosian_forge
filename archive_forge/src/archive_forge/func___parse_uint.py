import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
@staticmethod
def __parse_uint(parser, name, default):
    value = parser.get_optional(name, (int,))
    if value is None:
        value = default
    else:
        max_value = 2 ** 32 - 1
        if not 0 <= value <= max_value:
            raise error.Error('%s out of valid range 0 to %d' % (name, max_value), value)
    return value
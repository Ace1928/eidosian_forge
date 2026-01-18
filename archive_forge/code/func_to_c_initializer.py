import re
import uuid
import ovs.db.parser
from ovs.db import error
def to_c_initializer(uuid_, var):
    hex_string = uuid_.hex
    parts = ['0x%s' % hex_string[x * 8:(x + 1) * 8] for x in range(4)]
    return '{ %s },' % ', '.join(parts)
import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def to_lvalue_string(self):
    if self == StringType:
        return 's'
    return self.name
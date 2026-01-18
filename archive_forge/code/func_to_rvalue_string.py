import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def to_rvalue_string(self):
    if self == StringType:
        return 's->' + self.name
    return self.name
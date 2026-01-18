import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def to_const_c_type(self, prefix, refTable=True):
    nonconst = self.toCType(prefix, refTable)
    if '*' in nonconst:
        return 'const ' + nonconst
    else:
        return nonconst
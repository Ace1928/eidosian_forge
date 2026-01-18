import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def without_constraints(self):
    return BaseType(self.type)
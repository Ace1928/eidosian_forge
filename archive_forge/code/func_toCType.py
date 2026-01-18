import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def toCType(self, prefix, refTable=True):
    if self.ref_table_name:
        if not refTable:
            assert self.type == UuidType
            return 'struct uuid *'
        return 'struct %s%s *' % (prefix, self.ref_table_name.lower())
    else:
        return {IntegerType: 'int64_t ', RealType: 'double ', UuidType: 'struct uuid ', BooleanType: 'bool ', StringType: 'char *'}[self.type]
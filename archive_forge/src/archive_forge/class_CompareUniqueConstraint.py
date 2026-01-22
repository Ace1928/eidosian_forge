from itertools import zip_longest
from sqlalchemy import schema
from sqlalchemy.sql.elements import ClauseList
class CompareUniqueConstraint:

    def __init__(self, constraint):
        self.constraint = constraint

    def __eq__(self, other):
        r1 = isinstance(other, schema.UniqueConstraint) and self.constraint.name == other.name and (other.table.name == self.constraint.table.name) and (other.table.schema == self.constraint.table.schema)
        if not r1:
            return False
        for c1, c2 in zip_longest(self.constraint.columns, other.columns):
            if c1 is None and c2 is not None or (c2 is None and c1 is not None):
                return False
            if CompareColumn(c1) != c2:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
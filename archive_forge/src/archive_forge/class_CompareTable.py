from itertools import zip_longest
from sqlalchemy import schema
from sqlalchemy.sql.elements import ClauseList
class CompareTable:

    def __init__(self, table):
        self.table = table

    def __eq__(self, other):
        if self.table.name != other.name or self.table.schema != other.schema:
            return False
        for c1, c2 in zip_longest(self.table.c, other.c):
            if c1 is None and c2 is not None or (c2 is None and c1 is not None):
                return False
            if CompareColumn(c1) != c2:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
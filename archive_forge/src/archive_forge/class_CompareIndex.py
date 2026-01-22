from itertools import zip_longest
from sqlalchemy import schema
from sqlalchemy.sql.elements import ClauseList
class CompareIndex:

    def __init__(self, index, name_only=False):
        self.index = index
        self.name_only = name_only

    def __eq__(self, other):
        if self.name_only:
            return self.index.name == other.name
        else:
            return str(schema.CreateIndex(self.index)) == str(schema.CreateIndex(other)) and self.index.dialect_kwargs == other.dialect_kwargs

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        expr = ClauseList(*self.index.expressions)
        try:
            expr_str = expr.compile().string
        except Exception:
            expr_str = str(expr)
        return f'<CompareIndex {self.index.name}({expr_str})>'
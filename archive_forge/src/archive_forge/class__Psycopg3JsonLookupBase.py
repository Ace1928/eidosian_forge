import json
from peewee import *
from peewee import Expression
from peewee import Node
from peewee import NodeList
from playhouse.postgres_ext import ArrayField
from playhouse.postgres_ext import DateTimeTZField
from playhouse.postgres_ext import IndexedFieldMixin
from playhouse.postgres_ext import IntervalField
from playhouse.postgres_ext import Match
from playhouse.postgres_ext import TSVectorField
from playhouse.postgres_ext import _JsonLookupBase
class _Psycopg3JsonLookupBase(_JsonLookupBase):

    def concat(self, rhs):
        if not isinstance(rhs, Node):
            rhs = Jsonb(rhs)
        return Expression(self.as_json(True), OP.CONCAT, rhs)

    def contains(self, other):
        clone = self.as_json(True)
        if isinstance(other, (list, dict)):
            return Expression(clone, JSONB_CONTAINS, Jsonb(other))
        return Expression(clone, JSONB_EXISTS, other)